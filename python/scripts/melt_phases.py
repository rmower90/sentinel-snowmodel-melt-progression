# melt_phases.py


import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass
from tqdm.auto import tqdm

PHASE_CODES = {
    0: "dry",
    1: "moistening",
    2: "ripening",
    3: "output",
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _window_mask_from_daily_any(mask_3h, min_hours=48.0):
    s = mask_3h.to_series().astype(bool)
    if s.empty:
        return mask_3h
    tindex = s.index
    daily_any = s.resample("1D").sum() > 0
    groups = (daily_any != daily_any.shift()).cumsum()
    out = pd.Series(False, index=tindex)
    for _, seg in daily_any.groupby(groups):
        if not seg.iloc[0]:
            continue  # False segment
        d0, d1 = seg.index[0], seg.index[-1]
        in_days = (tindex >= d0) & (tindex < d1 + pd.Timedelta(days=1))
        mask_seg = in_days & s.values
        if not mask_seg.any():
            continue
        first = tindex[mask_seg].min()
        last  = tindex[mask_seg].max()
        duration = (last - first) / np.timedelta64(1, "h")
        if duration >= min_hours:
            out.loc[(tindex >= first) & (tindex <= last)] = True
    return xr.DataArray(out, coords=mask_3h.coords, dims=mask_3h.dims)


def _forward_window_max(da, window_hours=72.0):
    t = da["time"].to_index()
    if len(t) < 2:
        return da.copy()
    dt = (t[1] - t[0]) / np.timedelta64(1, "h")
    n  = int(np.ceil(window_hours / dt))
    rev = da[::-1]
    rev_max = rev.rolling(time=n, min_periods=1).max()
    return rev_max[::-1]


def _forward_window_any_bool(mask_da, window_hours=24.0):
    t = mask_da["time"].to_index()
    if len(t) < 2:
        return mask_da.copy()
    dt = (t[1] - t[0]) / np.timedelta64(1, "h")
    n  = int(np.ceil(window_hours / dt))
    rev = mask_da[::-1].astype(int)
    rev_any = rev.rolling(time=n, min_periods=1).max()
    return rev_any[::-1].astype(bool)


def _detect_output_mask(
    swe,
    runoff,
    swe_min_output=0.10,
    persistence_hours=72.0,
    future_window_hours=72.0,
    merge_dry_hours=48.0,
):
    runoff_filled = runoff.fillna(0)
    swe_dec = xr.zeros_like(swe, dtype=bool)
    if swe.size > 1:
        swe_dec[1:] = swe.diff("time") < 0
    rc = runoff_filled.cumsum("time")
    rc_fwd = _forward_window_max(rc, window_hours=future_window_hours)
    has_future = (rc_fwd - rc) > 0
    swe_ok = swe > swe_min_output
    candidate = swe_ok & swe_dec & has_future
    mask = _window_mask_from_daily_any(candidate, min_hours=persistence_hours)

    # merge short dry gaps between output
    s = mask.to_series().astype(bool)
    tindex = s.index
    groups = (s != s.shift()).cumsum()
    for _, seg in s.groupby(groups):
        if not seg.iloc[0]:
            gap_hours = (seg.index[-1] - seg.index[0]) / np.timedelta64(1, "h")
            start, end = seg.index[0], seg.index[-1]
            prev_out = (start != tindex[0]) and s.iloc[tindex.get_loc(start) - 1]
            next_out = (end   != tindex[-1]) and s.iloc[tindex.get_loc(end) + 1]
            if gap_hours <= merge_dry_hours and prev_out and next_out:
                # Only fill gap if SWE stays above threshold throughout
                gap_swe = swe.sel(time=seg.index)
                if (gap_swe >= swe_min_output).all():
                    s.loc[seg.index] = True

    return xr.DataArray(s.values, coords=mask.coords, dims=mask.dims)


def _detect_between_output_ripening(
    swe,
    ts,
    output_mask,
    max_gap_days=14.0,
    mean_thresh=-1.5,
    swe_thresh=0.10,
):
    tmean = ts.mean(dim="layer", skipna=True)
    s = output_mask.to_series().astype(bool)
    tindex = s.index
    groups = (s != s.shift()).cumsum()
    ripe = pd.Series(False, index=tindex)
    for _, seg in s.groupby(groups):
        if seg.iloc[0]:
            continue
        start, end = seg.index[0], seg.index[-1]
        dur_days = float((end - start) / np.timedelta64(1, "D"))
        if dur_days <= max_gap_days:
            swe_sub = swe.sel(time=slice(start, end))
            if (swe_sub >= swe_thresh).any():
                mean_sub = tmean.sel(time=slice(start, end))
                if np.nanmean(mean_sub.values) > mean_thresh:
                    ripe.loc[seg.index] = True
    return xr.DataArray(ripe.values, coords=output_mask.coords, dims=output_mask.dims)


def _internal_liq_mask(ds, liq_var="melt", water_thresh=1e-4):
    liq = ds[liq_var]
    valid = liq.notnull()
    layers = liq["layer"]
    last_valid_layer = (valid * layers).where(valid).max("layer").fillna(0)
    inner = layers < last_valid_layer
    return ((liq >= water_thresh) & inner).any("layer")


def _top_layer_liq_mask(ds, liq_var="melt", water_thresh=1e-4):
    liq = ds[liq_var]
    valid = liq.notnull()
    layers = liq["layer"]
    last_valid_layer = (valid * layers).where(valid).max("layer").fillna(0)
    top_mask = layers == last_valid_layer
    return ((liq >= water_thresh) & top_mask).any("layer")


def _detect_il_ripening(
    ds,
    output_mask,
    base_ripe_mask,
    int_liq_mask,
    mean_thresh=-1.5,
    temp_var="tsfc",
):
    time = ds["time"]
    t_index = time.to_index()
    tmean = ds[temp_var].mean(dim="layer", skipna=True)

    output_s    = output_mask.to_series().astype(bool)
    base_ripe_s = base_ripe_mask.to_series().astype(bool)
    int_s       = int_liq_mask.to_series().astype(bool)
    dry_s       = (~output_s) & (~base_ripe_s)

    tmean_s    = tmean.to_series()
    daily_mean = tmean_s.resample("1D").mean()
    daily_warm = daily_mean > mean_thresh
    daily_int  = int_s.resample("1D").sum() > 0

    il_ripe_s = pd.Series(False, index=t_index)

    groups = (dry_s != dry_s.shift()).cumsum()
    for _, seg in dry_s.groupby(groups):
        if not seg.iloc[0]:
            continue
        seg_times = seg.index
        seg_int   = int_s.loc[seg_times]
        seed_times = seg_int[seg_int].index

        for seed in seed_times:
            day = pd.Timestamp(seed.normalize())
            while True:
                day_start = day
                day_end   = day + pd.Timedelta(days=1)
                day_mask  = (seg_times >= day_start) & (seg_times < day_end)
                day_times = seg_times[day_mask]
                if len(day_times) == 0:
                    break
                il_ripe_s.loc[day_times] = True

                day = day + pd.Timedelta(days=1)
                day_start = day
                day_end   = day + pd.Timedelta(days=1)
                next_mask = (seg_times >= day_start) & (seg_times < day_end)
                if not next_mask.any():
                    break
                d_int  = bool(daily_int.get(day, False))
                d_warm = bool(daily_warm.get(day, False))
                if not (d_int or d_warm):
                    break

    il_ripe_da = xr.DataArray(il_ripe_s.values, coords={"time": time}, dims=("time",))
    il_ripe_da = il_ripe_da.where(dry_s.values, False)
    return il_ripe_da


def _detect_moistening(
    ds,
    output_mask,
    total_ripe_mask,
    swe_thresh=0.10,
    water_thresh=1e-4,
    min_moist_hours=24.0,
    liq_var="melt",
):
    """
    Moistening phase using daily-window episodes.
    """
    swe = ds["swed"]

    dry = (~output_mask) & (~total_ripe_mask)
    has_top_liq = _top_layer_liq_mask(ds, liq_var=liq_var, water_thresh=water_thresh)
    moist_candidates = dry & (swe >= swe_thresh) & has_top_liq

    moist_windows = _window_mask_from_daily_any(
        moist_candidates, min_hours=min_moist_hours
    )

    moist = moist_windows & dry
    return moist


# ---------- NEW: fill short dry gaps between phases ----------

def _fill_short_dry_gaps(phase_code_da, max_gap_hours=72.0):
    """
    Fill dry (0) segments whose length in timesteps is <= max_steps,
    where max_steps ~ max_gap_hours / dt.

    Neighbor rules for each short dry segment:
      - Build a set of nearest non-dry neighbors {prev_code, next_code}.
      - If 1 in neighbors -> fill gap as moistening (1).
      - Else if 2 or 3 in neighbors -> fill gap as ripening (2).
      - Else leave it as dry.
    """
    s = phase_code_da.to_series().astype("int8")
    if s.empty:
        return phase_code_da

    tindex = s.index

    # infer dt (hours) from first two timestamps
    if len(tindex) > 1:
        dt_hours = (tindex[1] - tindex[0]) / np.timedelta64(1, "h")
    else:
        dt_hours = 3.0  # fallback

    # max number of steps to treat as a "short" gap (inclusive)
    max_steps = int(round(max_gap_hours / dt_hours))

    groups = (s != s.shift()).cumsum()

    for _, seg in s.groupby(groups):
        code_here = seg.iloc[0]
        if code_here != 0:
            continue  # only process dry segments

        n_steps = len(seg)

        # NEW: step-based threshold, inclusive
        if n_steps > max_steps:
            continue  # too long to fill

        start_time = seg.index[0]
        end_time   = seg.index[-1]
        start_pos  = tindex.get_loc(start_time)
        end_pos    = tindex.get_loc(end_time)

        # nearest previous non-dry
        prev_code = None
        for j in range(start_pos - 1, -1, -1):
            if s.iloc[j] != 0:
                prev_code = int(s.iloc[j])
                break

        # nearest next non-dry
        next_code = None
        for j in range(end_pos + 1, len(s)):
            if s.iloc[j] != 0:
                next_code = int(s.iloc[j])
                break

        neighbors = {c for c in (prev_code, next_code) if c is not None}
        if not neighbors:
            continue  # isolated dry block, nothing to infer from

        new_code = None

        # Moist has highest priority if present on either side
        if 1 in neighbors:
            new_code = 1
        # Otherwise, any ripening or output neighbor -> ripening
        elif 2 in neighbors or 3 in neighbors:
            new_code = 2

        if new_code is not None:
            s.loc[seg.index] = new_code

    filled = xr.DataArray(
        s.values, coords=phase_code_da.coords, dims=phase_code_da.dims
    ).astype("int8")

    return filled



# ------------------------------------------------------------------
# Main classifier
# ------------------------------------------------------------------

def classify_snowmodel_melt_phases_3h_full(
    ds,
    liq_var="melt",
    swe_var="swed",
    runoff_var="roff",
    temp_var="tsfc",
    swe_min_output=0.10,
    water_thresh=1e-4,
    output_persistence_hours=72.0,
    output_future_hours=72.0,
    output_merge_dry_hours=48.0,
    ripen_gap_days=14.0,
    ripen_mean_temp_thresh=-1.5,
    moist_min_hours=24.0,
    dry_gap_hours=72.0,   # <--- gaps < 3 days will be filled by rules above
):
    swe = ds[swe_var]
    runoff = ds[runoff_var]
    ts = ds[temp_var]
    time = ds["time"]

    # ----- Output -----
    output_mask = _detect_output_mask(
        swe,
        runoff,
        swe_min_output=swe_min_output,
        persistence_hours=output_persistence_hours,
        future_window_hours=output_future_hours,
        merge_dry_hours=output_merge_dry_hours,
    )

    # ----- Between-output ripening -----
    between_ripe = _detect_between_output_ripening(
        swe,
        ts,
        output_mask,
        max_gap_days=ripen_gap_days,
        mean_thresh=ripen_mean_temp_thresh,
        swe_thresh=swe_min_output,
    )

    # ----- IL-driven ripening -----
    int_liq = _internal_liq_mask(ds, liq_var=liq_var, water_thresh=water_thresh)
    il_ripe = _detect_il_ripening(
        ds,
        output_mask,
        between_ripe,
        int_liq,
        mean_thresh=ripen_mean_temp_thresh,
        temp_var=temp_var,
    )
    total_ripe = between_ripe | il_ripe

    # ----- Moistening -----
    moist_mask = _detect_moistening(
        ds,
        output_mask,
        total_ripe,
        swe_thresh=swe_min_output,
        water_thresh=water_thresh,
        min_moist_hours=moist_min_hours,
        liq_var=liq_var,
    )

    # ----- Build initial phase_code (priority: output > ripening > moist > dry) -----
    phase_code = xr.zeros_like(swe, dtype="int8")  # 0 = dry

    # 3 = output
    phase_code = phase_code.where(~output_mask, 3)

    # 2 = ripening (between-output + IL), not already output, and only where SWE >= threshold
    ripening_mask = total_ripe & ~output_mask & (swe >= swe_min_output)
    phase_code = phase_code.where(~ripening_mask, 2)

    # 1 = moistening, not output or ripening
    moist_final = moist_mask & ~output_mask & ~ripening_mask
    phase_code = phase_code.where(~moist_final, 1)

    # ----- Fill short dry gaps (< dry_gap_hours) per your rules -----
    phase_code_filled = _fill_short_dry_gaps(phase_code, max_gap_hours=dry_gap_hours)

    # Map to names
    code_to_name = np.vectorize(PHASE_CODES.get, otypes=[object])
    phase_name = xr.apply_ufunc(
        code_to_name,
        phase_code_filled,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str],
    )

    out = xr.Dataset(
        data_vars=dict(
            swe=("time", swe.values),
            runoff=("time", runoff.values),
            phase_code_3h=("time", phase_code_filled.data),
            phase_name_3h=("time", phase_name.data),
        ),
        coords=dict(time=time),
    )
    return out


# ======================================================================
# REVAMP: classify_snowmodel_melt_phases_3h_full_revamp
# ======================================================================
#
# Key changes vs original:
#   1. Detection order: output -> moistening -> ripening (Colbeck sequence)
#   2. Output requires minimum runoff magnitude, not just > 0
#   3. Separate water thresholds for surface vs internal liquid
#   4. Daily any-occurrence expansion for moistening and ripening
#      (if condition met at ANY 3h timestep, full day is labeled)
# ======================================================================

def _detect_output_mask_v2(
    swe,
    runoff,
    swe_min_output=0.10,
    persistence_hours=72.0,
    future_window_hours=72.0,
    merge_dry_hours=48.0,
    min_runoff_rate=0.001,
):
    """
    Output detection with minimum runoff magnitude threshold.

    Change vs _detect_output_mask: requires cumulative runoff increase
    in the forward window to exceed min_runoff_rate (default 1 mm),
    not just > 0. Prevents trivial rain-on-snow or numerical noise
    from triggering output phase in mid-winter.
    """
    runoff_filled = runoff.fillna(0)
    swe_dec = xr.zeros_like(swe, dtype=bool)
    if swe.size > 1:
        swe_dec[1:] = swe.diff("time") < 0
    rc = runoff_filled.cumsum("time")
    rc_fwd = _forward_window_max(rc, window_hours=future_window_hours)
    has_future = (rc_fwd - rc) > min_runoff_rate   # <-- changed from > 0
    swe_ok = swe > swe_min_output
    candidate = swe_ok & swe_dec & has_future
    mask = _window_mask_from_daily_any(candidate, min_hours=persistence_hours)

    # merge short dry gaps between output
    s = mask.to_series().astype(bool)
    tindex = s.index
    groups = (s != s.shift()).cumsum()
    for _, seg in s.groupby(groups):
        if not seg.iloc[0]:
            gap_hours = (seg.index[-1] - seg.index[0]) / np.timedelta64(1, "h")
            start, end = seg.index[0], seg.index[-1]
            prev_out = (start != tindex[0]) and s.iloc[tindex.get_loc(start) - 1]
            next_out = (end   != tindex[-1]) and s.iloc[tindex.get_loc(end) + 1]
            if gap_hours <= merge_dry_hours and prev_out and next_out:
                gap_swe = swe.sel(time=seg.index)
                if (gap_swe >= swe_min_output).all():
                    s.loc[seg.index] = True

    return xr.DataArray(s.values, coords=mask.coords, dims=mask.dims)


def _detect_moistening_v2(
    ds,
    output_mask,
    swe_thresh=0.10,
    water_thresh_surface=1e-4,
    liq_var="melt",
):
    """
    Moistening detection — runs BEFORE ripening in the revamp pipeline.

    If top-layer liquid >= water_thresh_surface at ANY 3h timestep during
    a calendar day (and SWE >= swe_thresh, and not output), ALL timesteps
    of that day are labeled moistening. This captures diurnal melt-refreeze:
    surface wetting during warm hours followed by overnight refreeze is
    still a day in the moistening regime.
    """
    swe = ds["swed"]
    not_output = ~output_mask
    has_top_liq = _top_layer_liq_mask(ds, liq_var=liq_var, water_thresh=water_thresh_surface)
    swe_ok = swe >= swe_thresh
    candidate = not_output & swe_ok & has_top_liq

    # daily any-occurrence expansion
    s = candidate.to_series().astype(bool)
    daily_any = s.resample("1D").max()  # True if any 3h step that day
    # broadcast daily flag back to 3h index
    moist_daily = daily_any.reindex(s.index, method="ffill")
    moist = xr.DataArray(
        moist_daily.values & not_output.values,
        coords=output_mask.coords, dims=output_mask.dims,
    )
    return moist


def _detect_il_ripening_v2(
    ds,
    output_mask,
    moist_mask,
    base_ripe_mask,
    int_liq_mask,
):
    """
    Internal-liquid ripening with daily any-occurrence expansion.

    If ANY inner layer has liquid >= water_thresh at any 3h timestep
    during a calendar day (and the timestep is not output, moistening,
    or between-output ripening), ALL available timesteps of that day
    are labeled ripening. No forward-walk propagation — each day must
    independently show internal liquid to qualify.
    """
    time = ds["time"]
    t_index = time.to_index()

    output_s    = output_mask.to_series().astype(bool)
    moist_s     = moist_mask.to_series().astype(bool)
    base_ripe_s = base_ripe_mask.to_series().astype(bool)
    int_s       = int_liq_mask.to_series().astype(bool)
    available   = (~output_s) & (~moist_s) & (~base_ripe_s)

    # only consider internal liquid at available timesteps
    candidate = int_s & available
    daily_any = candidate.resample("1D").max()  # True if any 3h step that day
    # broadcast daily flag back to 3h index
    ripe_daily = daily_any.reindex(t_index, method="ffill")

    il_ripe_da = xr.DataArray(
        ripe_daily.values & available.values,
        coords={"time": time}, dims=("time",),
    )
    return il_ripe_da


def classify_snowmodel_melt_phases_3h_full_revamp(
    ds,
    liq_var="melt",
    swe_var="swed",
    runoff_var="roff",
    temp_var="tsfc",
    swe_min_output=0.10,
    water_thresh_surface=1e-4,
    water_thresh_internal=1e-4,
    min_runoff_rate=0.001,
    output_persistence_hours=72.0,
    output_future_hours=72.0,
    output_merge_dry_hours=48.0,
    ripen_gap_days=14.0,
    ripen_mean_temp_thresh=-1.5,
    dry_gap_hours=72.0,
):
    """
    Revamped melt-phase classifier.  Same input Dataset, same output Dataset.

    Key changes vs classify_snowmodel_melt_phases_3h_full:
      1. Detection order: output -> moistening -> ripening -> gap fill
         (moistening before ripening, matching Colbeck physical sequence)
      2. Output requires minimum runoff magnitude (min_runoff_rate),
         not just > 0, to suppress spurious mid-winter output
      3. Separate water_thresh for surface (moistening) vs internal (ripening)
      4. Daily any-occurrence expansion: if moistening/ripening condition
         is met at ANY 3h timestep in a day, the full day is labeled.
         Captures diurnal melt-refreeze as a single regime day.
    """
    swe = ds[swe_var]
    runoff = ds[runoff_var]
    ts = ds[temp_var]
    time = ds["time"]

    # ----- Stage 1: Output -----
    output_mask = _detect_output_mask_v2(
        swe,
        runoff,
        swe_min_output=swe_min_output,
        persistence_hours=output_persistence_hours,
        future_window_hours=output_future_hours,
        merge_dry_hours=output_merge_dry_hours,
        min_runoff_rate=min_runoff_rate,
    )

    # ----- Stage 2: Moistening (BEFORE ripening) -----
    moist_mask = _detect_moistening_v2(
        ds,
        output_mask,
        swe_thresh=swe_min_output,
        water_thresh_surface=water_thresh_surface,
        liq_var=liq_var,
    )

    # ----- Stage 3: Between-output ripening -----
    between_ripe = _detect_between_output_ripening(
        swe,
        ts,
        output_mask,
        max_gap_days=ripen_gap_days,
        mean_thresh=ripen_mean_temp_thresh,
        swe_thresh=swe_min_output,
    )

    # ----- Stage 4: IL-driven ripening -----
    int_liq = _internal_liq_mask(ds, liq_var=liq_var, water_thresh=water_thresh_internal)
    il_ripe = _detect_il_ripening_v2(
        ds,
        output_mask,
        moist_mask,
        between_ripe,
        int_liq,
    )
    total_ripe = between_ripe | il_ripe

    # ----- Build phase_code (priority: output > ripening > moistening > dry) -----
    phase_code = xr.zeros_like(swe, dtype="int8")  # 0 = dry

    # 3 = output
    phase_code = phase_code.where(~output_mask, 3)

    # 2 = ripening, not already output, SWE >= threshold
    ripening_mask = total_ripe & ~output_mask & (swe >= swe_min_output)
    phase_code = phase_code.where(~ripening_mask, 2)

    # 1 = moistening, not output or ripening
    moist_final = moist_mask & ~output_mask & ~ripening_mask
    phase_code = phase_code.where(~moist_final, 1)

    # ----- Stage 5: Fill short dry gaps -----
    phase_code_filled = _fill_short_dry_gaps(phase_code, max_gap_hours=dry_gap_hours)

    # Map to names
    code_to_name = np.vectorize(PHASE_CODES.get, otypes=[object])
    phase_name = xr.apply_ufunc(
        code_to_name,
        phase_code_filled,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str],
    )

    out = xr.Dataset(
        data_vars=dict(
            swe=("time", swe.values),
            runoff=("time", runoff.values),
            phase_code_3h=("time", phase_code_filled.data),
            phase_name_3h=("time", phase_name.data),
        ),
        coords=dict(time=time),
    )
    return out


# ──────────────────────────────────────────────────────────────────
# v3 – Temperature-augmented classifier
# ──────────────────────────────────────────────────────────────────

def _detect_moistening_v3(
    ds,
    output_mask,
    swe_thresh=0.10,
    water_thresh_surface=1e-4,
    temp_thresh=0.0,
    liq_var="melt",
    temp_var="tsfc",
):
    """
    Moistening with temperature criterion.

    A timestep is a moistening candidate if (not output, SWE >= thresh) AND:
      - top-layer liquid >= water_thresh_surface   OR
      - top-layer temperature >= temp_thresh (essentially 0°C)

    Daily any-occurrence expansion: if ANY 3h timestep in a calendar day
    is a candidate, the full day is moistening.
    """
    swe = ds["swed"]
    temp = ds[temp_var]
    liq = ds[liq_var]
    valid = liq.notnull()
    layers = liq["layer"]
    last_valid_layer = (valid * layers).where(valid).max("layer").fillna(0)
    top_mask = layers == last_valid_layer

    # top-layer liquid criterion
    has_top_liq = ((liq >= water_thresh_surface) & top_mask).any("layer")

    # top-layer temperature criterion
    top_temp = temp.where(top_mask).max("layer")  # single value per timestep
    has_top_warm = top_temp >= temp_thresh

    not_output = ~output_mask
    swe_ok = swe >= swe_thresh
    candidate = not_output & swe_ok & (has_top_liq | has_top_warm)

    # daily any-occurrence expansion
    s = candidate.to_series().astype(bool)
    daily_any = s.resample("1D").max()
    moist_daily = daily_any.reindex(s.index, method="ffill")
    moist = xr.DataArray(
        moist_daily.values & not_output.values,
        coords=output_mask.coords, dims=output_mask.dims,
    )
    return moist


def _detect_il_ripening_v3(
    ds,
    output_mask,
    moist_mask,
    base_ripe_mask,
    int_liq_mask,
    temp_thresh=0.0,
    majority_frac=0.5,
    temp_var="tsfc",
):
    """
    Internal-liquid ripening with temperature criterion.

    A timestep is a ripening candidate if (available) AND:
      - any inner layer has liquid >= water_thresh_internal   OR
      - fraction of valid layers with T >= temp_thresh is >= majority_frac

    Daily any-occurrence expansion: if ANY 3h timestep in a calendar day
    is a candidate, the full day is ripening.
    """
    time = ds["time"]
    t_index = time.to_index()
    temp = ds[temp_var]

    # fraction of valid layers at ~0°C per timestep
    valid = temp.notnull()
    n_valid = valid.sum("layer")
    n_warm = (temp >= temp_thresh).where(valid).sum("layer")
    frac_warm = (n_warm / n_valid.where(n_valid > 0)).fillna(0)
    has_majority_warm = frac_warm >= majority_frac

    output_s    = output_mask.to_series().astype(bool)
    moist_s     = moist_mask.to_series().astype(bool)
    base_ripe_s = base_ripe_mask.to_series().astype(bool)
    int_s       = int_liq_mask.to_series().astype(bool)
    warm_s      = has_majority_warm.to_series().astype(bool)
    available   = (~output_s) & (~moist_s) & (~base_ripe_s)

    candidate = (int_s | warm_s) & available
    daily_any = candidate.resample("1D").max()
    ripe_daily = daily_any.reindex(t_index, method="ffill")

    il_ripe_da = xr.DataArray(
        ripe_daily.values & available.values,
        coords={"time": time}, dims=("time",),
    )
    return il_ripe_da


def classify_snowmodel_melt_phases_3h_full_temp(
    ds,
    liq_var="melt",
    swe_var="swed",
    runoff_var="roff",
    temp_var="tsfc",
    swe_min_output=0.10,
    water_thresh_surface=1e-4,
    water_thresh_internal=1e-4,
    min_runoff_rate=0.001,
    temp_thresh=0.0,
    majority_frac=0.5,
    output_persistence_hours=72.0,
    output_future_hours=72.0,
    output_merge_dry_hours=48.0,
    ripen_gap_days=14.0,
    ripen_mean_temp_thresh=-1.5,
    dry_gap_hours=72.0,
):
    """
    Temperature-augmented melt-phase classifier.

    Same pipeline as classify_snowmodel_melt_phases_3h_full_revamp, but
    moistening and ripening detection are expanded with temperature criteria:

      - Moistening: top-layer liquid >= water_thresh_surface
                    OR top-layer T >= temp_thresh
      - Ripening:   any inner layer liquid >= water_thresh_internal
                    OR fraction of valid layers with T >= temp_thresh >= majority_frac

    Parameters
    ----------
    temp_thresh : float
        Temperature threshold for "essentially 0°C".  Default 0.0.
        The model clamps layer temperatures at ~0.01°C at the melting point;
        0.0 captures this isothermal state.
    majority_frac : float
        Fraction of valid layers that must have T >= temp_thresh to trigger
        temperature-based ripening.  Default 0.5 (majority).
    """
    swe = ds[swe_var]
    runoff = ds[runoff_var]
    ts = ds[temp_var]
    time = ds["time"]

    # ----- Stage 1: Output -----
    output_mask = _detect_output_mask_v2(
        swe,
        runoff,
        swe_min_output=swe_min_output,
        persistence_hours=output_persistence_hours,
        future_window_hours=output_future_hours,
        merge_dry_hours=output_merge_dry_hours,
        min_runoff_rate=min_runoff_rate,
    )

    # ----- Stage 2: Moistening (temp-augmented) -----
    moist_mask = _detect_moistening_v3(
        ds,
        output_mask,
        swe_thresh=swe_min_output,
        water_thresh_surface=water_thresh_surface,
        temp_thresh=temp_thresh,
        liq_var=liq_var,
        temp_var=temp_var,
    )

    # ----- Stage 3: Between-output ripening -----
    between_ripe = _detect_between_output_ripening(
        swe,
        ts,
        output_mask,
        max_gap_days=ripen_gap_days,
        mean_thresh=ripen_mean_temp_thresh,
        swe_thresh=swe_min_output,
    )

    # ----- Stage 4: IL-driven ripening (temp-augmented) -----
    int_liq = _internal_liq_mask(ds, liq_var=liq_var, water_thresh=water_thresh_internal)
    il_ripe = _detect_il_ripening_v3(
        ds,
        output_mask,
        moist_mask,
        between_ripe,
        int_liq,
        temp_thresh=temp_thresh,
        majority_frac=majority_frac,
        temp_var=temp_var,
    )
    total_ripe = between_ripe | il_ripe

    # ----- Build phase_code (priority: output > ripening > moistening > dry) -----
    phase_code = xr.zeros_like(swe, dtype="int8")  # 0 = dry

    # 3 = output
    phase_code = phase_code.where(~output_mask, 3)

    # 2 = ripening, not already output, SWE >= threshold
    ripening_mask = total_ripe & ~output_mask & (swe >= swe_min_output)
    phase_code = phase_code.where(~ripening_mask, 2)

    # 1 = moistening, not output or ripening
    moist_final = moist_mask & ~output_mask & ~ripening_mask
    phase_code = phase_code.where(~moist_final, 1)

    # ----- Stage 5: Fill short dry gaps -----
    phase_code_filled = _fill_short_dry_gaps(phase_code, max_gap_hours=dry_gap_hours)

    # Map to names
    code_to_name = np.vectorize(PHASE_CODES.get, otypes=[object])
    phase_name = xr.apply_ufunc(
        code_to_name,
        phase_code_filled,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str],
    )

    out = xr.Dataset(
        data_vars=dict(
            swe=("time", swe.values),
            runoff=("time", runoff.values),
            phase_code_3h=("time", phase_code_filled.data),
            phase_name_3h=("time", phase_name.data),
        ),
        coords=dict(time=time),
    )
    return out


def wet_snow_by_orbit(path,
                      cues_y,
                      cues_x,
                      orbit_,
                      showOutput = False):
    # Constants to be used in the weights calculation
    k = 0.5
    theta1 = 20
    theta2 = 45

    binary_lst = []
    for file in sorted(os.listdir(path)):
        if showOutput: print(file)
        ## load and process sentinel ##
        sentinel_ds = xr.open_zarr(path + file)
        sentinel_ds = sentinel_ds.rio.write_crs("EPSG:32611")

        W = xr.where(sentinel_ds.local_incidence_angle<theta1,1,k*(1 + ((theta2-sentinel_ds.local_incidence_angle)/(theta2-theta1)) ) )
        W = xr.where(sentinel_ds.local_incidence_angle>theta2,k,W)
        Rc = W*sentinel_ds.ratio_images.sel(band='vh').groupby('sat:relative_orbit') + (1-W)*sentinel_ds.ratio_images.sel(band='vv').groupby('sat:relative_orbit') 
        Rc = Rc.rio.write_crs(sentinel_ds.rio.crs)
        binary_wet_snow = xr.where(Rc<-2,1,0)
        orbit_slice = binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True) \
                          .expand_dims({'Time':len(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time)}) \
                          .assign_coords({'Time':(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time.values)})
        binary_lst.append(orbit_slice)

    wet_snow_tseries = xr.concat(binary_lst,dim = 'Time').drop_duplicates(dim = 'Time')
    
    idy_mammoth,idx_mammoth = sm_intersect_rectilinear(wet_snow_tseries.y.values,wet_snow_tseries.x.values,cues_y,cues_x)

    da_tseries = wet_snow_tseries.isel({'y':idy_mammoth,'x':idx_mammoth})[:,0]
    da_tseries.load()

    # melt_time = da_tseries.where(da_tseries == 1,drop = True)[0].Time.values
    melt_time = da_tseries.where(da_tseries == 1,drop = True).Time.values

    return wet_snow_tseries,da_tseries,melt_time

def weighted_wet_snow_by_orbit(path,
                      cues_y,
                      cues_x,
                      orbit_,
                      showOutput = False):
    # Constants to be used in the weights calculation
    k = 0.5
    theta1 = 20
    theta2 = 45

    binary_lst = []
    for file in sorted(os.listdir(path)):
        if showOutput: print(file)
        ## load and process sentinel ##
        sentinel_ds = xr.open_zarr(path + file)
        sentinel_ds = sentinel_ds.rio.write_crs("EPSG:32611")

        W = xr.where(sentinel_ds.local_incidence_angle<theta1,1,k*(1 + ((theta2-sentinel_ds.local_incidence_angle)/(theta2-theta1)) ) )
        W = xr.where(sentinel_ds.local_incidence_angle>theta2,k,W)
        Rc = W*sentinel_ds.ratio_images.sel(band='vh').groupby('sat:relative_orbit') + (1-W)*sentinel_ds.ratio_images.sel(band='vv').groupby('sat:relative_orbit') 
        Rc = Rc.rio.write_crs(sentinel_ds.rio.crs)
        binary_wet_snow = xr.where(Rc<-2,1,0)
        orbit_slice = binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True) \
                          .expand_dims({'Time':len(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time)}) \
                          .assign_coords({'Time':(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time.values)})
        binary_lst.append(orbit_slice)

    wet_snow_tseries = xr.concat(binary_lst,dim = 'Time').drop_duplicates(dim = 'Time')
    
    idy_mammoth,idx_mammoth = sm_intersect_rectilinear(wet_snow_tseries.y.values,wet_snow_tseries.x.values,cues_y,cues_x)

    da_tseries = wet_snow_tseries.isel({'y':idy_mammoth,'x':idx_mammoth})[:,0]
    da_tseries.load()

    # melt_time = da_tseries.where(da_tseries == 1,drop = True)[0].Time.values
    melt_time = da_tseries.where(da_tseries == 1,drop = True).Time.values

    return wet_snow_tseries,da_tseries,melt_time

def wet_snow_all_orbits(path,
                      cues_y,
                      cues_x,
                      showOutput = False):
    # Constants to be used in the weights calculation
    k = 0.5
    theta1 = 20
    theta2 = 45

    binary_lst = []
    for file in sorted(os.listdir(path)):
        if showOutput: print(file)
        ## load and process sentinel ##
        sentinel_ds = xr.open_zarr(path + file)
        sentinel_ds = sentinel_ds.rio.write_crs("EPSG:32611")

        W = xr.where(sentinel_ds.local_incidence_angle<theta1,1,k*(1 + ((theta2-sentinel_ds.local_incidence_angle)/(theta2-theta1)) ) )
        W = xr.where(sentinel_ds.local_incidence_angle>theta2,k,W)
        Rc = W*sentinel_ds.ratio_images.sel(band='vh').groupby('sat:relative_orbit') + (1-W)*sentinel_ds.ratio_images.sel(band='vv').groupby('sat:relative_orbit') 
        Rc = Rc.rio.write_crs(sentinel_ds.rio.crs)
        binary_wet_snow = xr.where(Rc<-2,1,0)
        binary_wet_snow = binary_wet_snow.expand_dims({'Time':len(binary_wet_snow.time)}) \
                                         .assign_coords({'Time':binary_wet_snow.time.values})
        # orbit_slice = binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True) \
        #                   .expand_dims({'Time':len(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time)}) \
        #                   .assign_coords({'Time':(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time.values)})
        binary_lst.append(binary_wet_snow)

    wet_snow_tseries = xr.concat(binary_lst,dim = 'Time').drop_duplicates(dim = 'Time')
    
    idy_mammoth,idx_mammoth = sm_intersect_rectilinear(wet_snow_tseries.y.values,wet_snow_tseries.x.values,cues_y,cues_x)

    da_tseries = wet_snow_tseries.isel({'y':idy_mammoth,'x':idx_mammoth})[:,0]
    da_tseries.load()
    

    # # melt_time = da_tseries.where(da_tseries == 1,drop = True)[0].Time.values
    melt_time = da_tseries.where(da_tseries == 1,drop = True).Time.values

    return wet_snow_tseries,da_tseries,melt_time

def sm_intersect_rectilinear(xlat,xlon,obs_lat,obs_lon):
    """
    Intersects lat and lon location of insitu observation with SnowModel grid
    Input:
      xlat - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      xlon - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      obs_lat - python float of latitude for observation.
      obs_lon - python float of longitude for observation.

    Output:
      idy - integer representing y-index of intersected point on SnowModel grid.
      idx - integer representing x-index of intersected point on SnowModel grid.
    """
    dist = np.sqrt((xlat - obs_lat)**2)
    idy = np.argwhere(dist == np.min(dist))[0][0]
    dist = np.sqrt((xlon - obs_lon)**2)
    idx = np.argwhere(dist == np.min(dist))[0][0]

    return idy,idx

def subset_orbit_band_space_time(ds,
                                 orbit,
                                 band,
                                 idy_,
                                 idx_,
                                ):
    
    return ds.where(ds.band == band,drop = True) \
             .where(ds['sat:relative_orbit'] == orbit,drop = True) \
             .where(ds.time >= np.datetime64(f'{int(ds.time[-1].dt.year)}-01-01')) \
             .isel({'y':idy_,'x':idx_})

def subset_orbit_band_space(ds,
                            orbit,
                            band,
                            idy_,
                            idx_,
                                ):
    return ds.where(ds.band == band,drop = True) \
             .where(ds['sat:relative_orbit'] == orbit,drop = True) \
             .isel({'y':idy_,'x':idx_})

def subset_orbit_band_time(ds,
                           orbit,
                           band,
                            ):
    return ds.where(ds.band == band,drop = True) \
             .where(ds['sat:relative_orbit'] == orbit,drop = True) \
             .where(ds.time >= np.datetime64(f'{int(ds.time[-1].dt.year)}-01-01'))

def subset_orbit_band(ds,
                      orbit,
                      band,
                      ):
    return ds.where(ds.band == band,drop = True) \
             .where(ds['sat:relative_orbit'] == orbit,drop = True) 


def s1_bsc_ref(ds_bsc,
               ds_ref,
               thresh = -2,
               idy_ = None,
               idx_ = None,
               showPlot = True,
               figsize = None,
               savePlot = False,
               figname = None,
              ):
    
    if idy_ is not None:
        ref_64 = subset_orbit_band_space(ds_ref,64,'vv',idy_,idx_)
        ref_137 = subset_orbit_band_space(ds_ref,137,'vv',idy_,idx_)
        ref_144 = subset_orbit_band_space(ds_ref,144,'vv',idy_,idx_)
        bsc_64 = subset_orbit_band_space_time(ds_bsc,64,'vv',idy_,idx_)
        bsc_137 = subset_orbit_band_space_time(ds_bsc,137,'vv',idy_,idx_)
        bsc_144 = subset_orbit_band_space_time(ds_bsc,144,'vv',idy_,idx_)
    else:
        ref_64 = subset_orbit_band(ds_ref,64,'vv')
        ref_137 = subset_orbit_band(ds_ref,137,'vv')
        ref_144 = subset_orbit_band(ds_ref,144,'vv')
        bsc_64 = subset_orbit_band_time(ds_bsc,64,'vv')
        bsc_137 = subset_orbit_band_time(ds_bsc,137,'vv')
        bsc_144 = subset_orbit_band_time(ds_bsc,144,'vv')
        

    # convert to dB
    bsc_64['bscatter'] = (10*np.log10(bsc_64['bscatter']))
    bsc_137['bscatter'] = (10*np.log10(bsc_137['bscatter']))
    bsc_144['bscatter'] = (10*np.log10(bsc_144['bscatter']))
    ref_64['reference_bscatter'] = (10*np.log10(ref_64['reference_bscatter']))
    ref_137['reference_bscatter'] = (10*np.log10(ref_137['reference_bscatter']))
    ref_144['reference_bscatter'] = (10*np.log10(ref_144['reference_bscatter']))

    # thresh.
    wet_64 = (bsc_64['bscatter'] - ref_64['reference_bscatter']) < thresh
    wet_137 = (bsc_137['bscatter'] - ref_137['reference_bscatter']) < thresh
    wet_144 = (bsc_144['bscatter'] - ref_144['reference_bscatter']) < thresh
    
    if showPlot:
        fig,ax = plt.subplots(2,1,sharex = True,figsize=figsize) 
        bsc_64.bscatter.plot(ax=ax[0],color = 'C0',label = '64')
        ax[0].axhline(ref_64['reference_bscatter'],color = 'C0',linestyle = '--')
        ax[0].scatter(bsc_64['bscatter'][:,0].where(wet_64[:,0,0]).time,
            bsc_64['bscatter'][:,0].where(wet_64[:,0,0]),marker = 'x',color = 'red')
        ax[0].set_xlim(np.datetime64(f'{int(bsc_64.time[-1].dt.year)}-01-01'),None)
        bsc_137.bscatter.plot(ax=ax[0],color = 'C1',label = '137')
        ax[0].axhline(ref_137['reference_bscatter'],color = 'C1',linestyle = '--')
        ax[0].scatter(bsc_137['bscatter'][:,0].where(wet_137[:,0,0]).time,
            bsc_137['bscatter'][:,0].where(wet_137[:,0,0]),marker = 'x',color = 'red')
        ax[0].set_title('Afternoon - 64 and 137 Orbits') 
        ax[0].legend()
        
        bsc_144.bscatter.plot(ax=ax[1],color = 'C2',label = '144')
        ax[1].axhline(ref_144['reference_bscatter'],color = 'C2',linestyle = '--')
        ax[1].scatter(bsc_144['bscatter'][:,0].where(wet_144[:,0,0]).time,
            bsc_144['bscatter'][:,0].where(wet_144[:,0,0]),marker = 'x',color = 'red')
        ax[1].set_title('Morning - 144 Orbit')
        ax[1].legend()
        plt.tight_layout()
        if savePlot:
            plt.savefig(f'./pngs/{figname}.png',dpi = 300)
        plt.show()
    return wet_64,wet_137,wet_144,bsc_64['bscatter'][:,0],bsc_137['bscatter'][:,0],bsc_144['bscatter'][:,0],ref_64,ref_137,ref_144

PHASE = xr.DataArray(
    np.array(["dry", "moist", "ripe", "output"], dtype=object),
    dims=["phase_id"],
)

def _to_1d_bool(da: xr.DataArray) -> xr.DataArray:
    """Squeeze to a 1D (time,) boolean DataArray."""
    if "time" not in da.dims:
        raise ValueError("Expected a DataArray with a 'time' dimension.")
    # squeeze any singleton dims like band, sat:relative_orbit
    da1 = da.squeeze(drop=True)
    # ensure boolean
    return da1.astype(bool)

def _min_time(vv: xr.DataArray) -> pd.Timestamp:
    vv1 = vv.squeeze(drop=True)
    tmin = vv1.idxmin("time").item()
    return pd.Timestamp(tmin)

def mean_min_time(vv_by_orbit: dict) -> pd.Timestamp:
    """
    vv_by_orbit: dict like {"pm_137": vv_da, "pm_64": vv_da, "am_144": vv_da, ...}
    Returns mean timestamp of minima across provided series.
    """
    mins = [ _min_time(vv) for vv in vv_by_orbit.values() ]
    # mean in nanoseconds since epoch
    mean_ns = int(np.mean([m.value for m in mins]))
    return pd.to_datetime(mean_ns)



def build_phase_intervals_ripe_start_both_stop_either_dry(
    wet_pm: xr.DataArray,             # afternoon wet flag (e.g. orbit 137)
    wet_am: Optional[xr.DataArray],   # morning wet flag (e.g. orbit 144) - can be None
    output_start: pd.Timestamp,       # mean minima time across orbits
    allow_ripe_end_two_dry: bool = False,  # kept for API symmetry; default False for immediate stop
) -> xr.Dataset:
    """
    Variant where RIPENING:
      - can only START when BOTH AM and PM are wet
      - PERSISTS only while BOTH remain wet
      - STOPS when either AM or PM becomes dry (immediate by default)

    Phases apply to [t_i, t_{i+1}) intervals on the union acquisition grid.
    """

    wet_pm = _to_1d_bool(wet_pm)
    times = pd.to_datetime(wet_pm["time"].values)

    if wet_am is not None:
        wet_am = _to_1d_bool(wet_am)
        times = times.union(pd.to_datetime(wet_am["time"].values))

    times = pd.DatetimeIndex(times).sort_values()
    if len(times) < 2:
        raise ValueError("Need at least 2 unique acquisition times to form intervals.")

    # Reindex WITH forward-fill: propagate wetness across gaps between orbits
    # This allows ripening to persist when one orbit has observations while the other has a gap
    wet_pm_series = wet_pm.to_pandas()
    wet_pm_u = xr.DataArray(
        wet_pm_series.reindex(times).ffill().values,
        dims=['time'],
        coords={'time': times}
    )
    wet_am_u = None
    if wet_am is not None:
        wet_am_series = wet_am.to_pandas()
        wet_am_u = xr.DataArray(
            wet_am_series.reindex(times).ffill().values,
            dims=['time'],
            coords={'time': times}
        )

    t0 = times[:-1]
    t1 = times[1:]

    in_ripe = False
    consecutive_any_dry = 0  # counts intervals where (am_wet or pm_wet) is False

    phase_id = np.zeros(len(t0), dtype=np.int16)  # default dry=0

    for i, ti in enumerate(t0):
        ti_ts = pd.Timestamp(ti)

        # Output rule
        if ti_ts >= output_start:
            phase_id[i] = 3
            continue

        pm_val = wet_pm_u.sel(time=ti).values
        # Treat NaN (no observation) as dry/no evidence of wetness
        pm_wet = False if pd.isna(pm_val) else bool(pm_val)

        # If no AM, can't classify moist/ripe
        if wet_am_u is None:
            phase_id[i] = 0
            continue

        am_val = wet_am_u.sel(time=ti).values
        # Treat NaN (no observation) as dry/no evidence of wetness
        am_wet = False if pd.isna(am_val) else bool(am_val)

        # Moistening: PM wet but AM not wet (only if not already ripe)
        if pm_wet and (not am_wet) and (not in_ripe):
            phase_id[i] = 1
            continue

        # Ripening start: require BOTH
        if (not in_ripe) and am_wet and pm_wet:
            in_ripe = True
            consecutive_any_dry = 0

        if in_ripe:
            both_wet = am_wet and pm_wet
            if both_wet:
                consecutive_any_dry = 0
                phase_id[i] = 2  # ripe
            else:
                consecutive_any_dry += 1

                # Immediate stop if either dry (default behavior)
                if (not allow_ripe_end_two_dry) or (consecutive_any_dry >= 2):
                    in_ripe = False
                    # phase_id[i] = 0  # back to dry (could choose moist if pm_wet & ~am_wet)
                    phase_id[i] = 1 if (pm_wet and (not am_wet)) else 0

                else:
                    phase_id[i] = 2  # grace interval (if you ever enable it)
            continue

        phase_id[i] = 0  # dry

    ds_out = xr.Dataset(
        {
            "phase_id": xr.DataArray(
                phase_id,
                dims=["interval"],
                coords={"interval_start": ("interval", t0)},
            ),
            "interval_end": xr.DataArray(
                pd.to_datetime(t1),
                dims=["interval"],
                coords={"interval_start": ("interval", t0)},
            ),
        }
    )

    ds_out["phase_name"] = PHASE.sel(phase_id=ds_out["phase_id"]).rename("phase_name")
    ds_out = ds_out.set_coords(["interval_end"])
    return ds_out


def build_phase_intervals_ripe_start_both_persist_am(
    wet_pm: xr.DataArray,             # afternoon wet flag (e.g. orbit 137)
    wet_am: Optional[xr.DataArray],   # morning wet flag (e.g. orbit 144) - can be None
    output_start: pd.Timestamp,       # mean minima time across orbits
    allow_ripe_end_two_dry: bool = True,
) -> xr.Dataset:
    """
    Variant where RIPENING can only START when BOTH AM and PM are wet,
    but once started it PERSISTS based only on AM wetness (until output or AM dry rule).

    Phases apply to [t_i, t_{i+1}) intervals on the union acquisition grid.
    """

    wet_pm = _to_1d_bool(wet_pm)
    times = pd.to_datetime(wet_pm["time"].values)

    if wet_am is not None:
        wet_am = _to_1d_bool(wet_am)
        times = times.union(pd.to_datetime(wet_am["time"].values))

    times = pd.DatetimeIndex(times).sort_values()
    if len(times) < 2:
        raise ValueError("Need at least 2 unique acquisition times to form intervals.")

    # Reindex WITH forward-fill: propagate wetness across gaps between orbits
    # This allows ripening to persist when one orbit has observations while the other has a gap
    wet_pm_series = wet_pm.to_pandas()
    wet_pm_u = xr.DataArray(
        wet_pm_series.reindex(times).ffill().values,
        dims=['time'],
        coords={'time': times}
    )
    wet_am_u = None
    if wet_am is not None:
        wet_am_series = wet_am.to_pandas()
        wet_am_u = xr.DataArray(
            wet_am_series.reindex(times).ffill().values,
            dims=['time'],
            coords={'time': times}
        )

    # Label intervals [t_i, t_{i+1}) using wet state at t_i
    t0 = times[:-1]
    t1 = times[1:]

    in_ripe = False
    consecutive_am_dry = 0

    phase_id = np.zeros(len(t0), dtype=np.int16)  # default dry=0

    for i, ti in enumerate(t0):
        ti_ts = pd.Timestamp(ti)

        # Rule 1: output starts at output_start
        if ti_ts >= output_start:
            phase_id[i] = 3  # output
            continue

        pm_val = wet_pm_u.sel(time=ti).values
        # Treat NaN (no observation) as dry/no evidence of wetness
        pm_wet = False if pd.isna(pm_val) else bool(pm_val)

        # If no AM series, cannot define moist/ripe with this framework
        if wet_am_u is None:
            phase_id[i] = 0
            continue

        am_val = wet_am_u.sel(time=ti).values
        # Treat NaN (no observation) as dry/no evidence of wetness
        am_wet = False if pd.isna(am_val) else bool(am_val)

        # Rule 2: moistening = PM wet but AM not wet (only if not already ripe)
        if pm_wet and (not am_wet) and (not in_ripe):
            phase_id[i] = 1  # moist
            continue

        # Rule 3: ripe initiation requires BOTH AM and PM wet
        if (not in_ripe) and am_wet and pm_wet:
            in_ripe = True
            consecutive_am_dry = 0

        # Rule 4: ripe persistence depends ONLY on AM wetness
        if in_ripe:
            if am_wet:
                consecutive_am_dry = 0
                phase_id[i] = 2  # ripe
            else:
                consecutive_am_dry += 1
                if allow_ripe_end_two_dry and (consecutive_am_dry >= 2):
                    in_ripe = False
                    phase_id[i] = 0  # back to dry
                else:
                    phase_id[i] = 2  # still ripe during grace period
            continue

        # Otherwise dry
        phase_id[i] = 0

    ds_out = xr.Dataset(
        {
            "phase_id": xr.DataArray(
                phase_id,
                dims=["interval"],
                coords={"interval_start": ("interval", t0)},
            ),
            "interval_end": xr.DataArray(
                pd.to_datetime(t1),
                dims=["interval"],
                coords={"interval_start": ("interval", t0)},
            ),
        }
    )

    ds_out["phase_name"] = PHASE.sel(phase_id=ds_out["phase_id"]).rename("phase_name")
    ds_out = ds_out.set_coords(["interval_end"])
    return ds_out


def first_sustained_true(mask: xr.DataArray, window: int = 3, min_count: int = 2):
    """
    mask: (Time,) boolean DataArray (True/False/NaN)
    Returns: index (int) of first time i where sum(mask[i:i+window]) >= min_count.
             None if never satisfied.
    """
    m = mask.fillna(False).astype(int).values
    n = len(m)
    for i in range(n):
        if m[i:i+window].sum() >= min_count:
            return i
    return None

def _effective_wet_on_union_time(wet_da_bool, union_time, window_days=10):
    """
    wet_da_bool: xr.DataArray(Time,) bool on acquisition times
    union_time:  xr.DataArray(Time,) datetime64[ns] (the union grid you classify on)
    Returns np.ndarray[bool] aligned to union_time: True if any wet detection within ±window_days.
    """
    s = _da_bool_to_series(wet_da_bool)
    wet_days = pd.to_datetime(s[s].index).normalize()

    ut = pd.to_datetime(union_time.values).normalize()
    out = np.zeros(len(ut), dtype=bool)

    # vector-ish: for each wet day, flag union times within window
    win = pd.Timedelta(days=window_days)
    for d in wet_days:
        out |= (np.abs(ut - d) <= win)

    return out


def _times_where_true(da_bool: xr.DataArray) -> pd.DatetimeIndex:
    """Return acquisition timestamps where da_bool is True."""
    t = pd.to_datetime(da_bool["Time"].values)
    v = np.asarray(da_bool.values).astype(bool)
    return pd.DatetimeIndex(t[v])

def _expand_times_to_daily_true(
    true_times: pd.DatetimeIndex,
    daily_index: pd.DatetimeIndex,
    window_days: int,
) -> pd.Series:
    """
    Mark a daily boolean series True for any day within +/- window_days of any true_time.
    """
    out = pd.Series(False, index=daily_index)
    if len(true_times) == 0:
        return out

    # normalize to days
    true_days = pd.DatetimeIndex(true_times.normalize().unique()).sort_values()

    # expand each true day by +/- window_days
    for d in true_days:
        lo = d - pd.Timedelta(days=window_days)
        hi = d + pd.Timedelta(days=window_days)
        out.loc[(out.index >= lo) & (out.index <= hi)] = True
    return out

def _first_sustained_run_day(series_bool: pd.Series, sustain_days: int):
    """Return first day of a >= sustain_days consecutive True run, else NaT."""
    if series_bool.sum() == 0:
        return pd.NaT
    # run-length encoding on daily bool
    x = series_bool.values.astype(int)
    # find starts of runs
    starts = np.where((x == 1) & (np.r_[0, x[:-1]] == 0))[0]
    for s in starts:
        # run length
        e = s
        while e < len(x) and x[e] == 1:
            e += 1
        if (e - s) >= sustain_days:
            return series_bool.index[s]
    return pd.NaT


def _end_of_longest_true_run_day(series_bool: pd.Series):
    """
    Return the last day (Timestamp) of the longest consecutive True run in a daily boolean series.
    Returns pd.NaT if there are no True values.
    """
    if series_bool is None or series_bool.sum() == 0:
        return pd.NaT

    x = series_bool.values.astype(bool)
    starts = np.where(x & ~np.r_[False, x[:-1]])[0]

    best_len = 0
    best_end_idx = None

    for s in starts:
        e = s
        while e < len(x) and x[e]:
            e += 1
        run_len = e - s
        if run_len > best_len:
            best_len = run_len
            best_end_idx = e - 1  # inclusive

    return series_bool.index[best_end_idx] if best_end_idx is not None else pd.NaT

# def classify_sentinel_melt_phases_weighted(
#     melt_da: xr.DataArray,
#     morning_state: str = "descending",
#     afternoon_state: str = "ascending",
#     output_window: int = 3,            # legacy, unused
#     output_min_count: int = 2,         # legacy, unused
#     output_requires_ripe: bool = True, # kept for compatibility (unused in Option 1)
#     output_window_days: int = 10,      # +/- days expansion around wet acquisitions
#     output_sustain_days: int = 10,     # legacy, unused in Option 1
#     **kwargs,
# ):
#     PHASE_CODES = {0: "dry", 1: "moistening", 2: "ripening", 3: "output"}

#     # --- PM/AM wet on union acquisition grid (bool) ---
#     aft_da  = melt_da.where(melt_da["sat:orbit_state"] == afternoon_state, drop=True)
#     morn_da = melt_da.where(melt_da["sat:orbit_state"] == morning_state, drop=True)

#     aft_wet_u  = (aft_da == 1).any("sat:relative_orbit").fillna(False).astype(bool)
#     morn_wet_u = (morn_da == 1).any("sat:relative_orbit").fillna(False).astype(bool)

#     time_u = aft_wet_u["Time"].astype("datetime64[ns]")
#     t_u = pd.to_datetime(time_u.values)

#     # --- build daily axis ---
#     daily = pd.date_range(t_u.min().normalize(), t_u.max().normalize(), freq="D")

#     # --- expand wet acquisitions to "effective wet days" ---
#     aft_true_times  = _times_where_true(aft_wet_u)
#     morn_true_times = _times_where_true(morn_wet_u)

#     aft_eff  = _expand_times_to_daily_true(aft_true_times,  daily, window_days=output_window_days)
#     morn_eff = _expand_times_to_daily_true(morn_true_times, daily, window_days=output_window_days)

#     # --- daily phase logic ---
#     daily_moist = aft_eff & (~morn_eff)
#     daily_ripe  = aft_eff & morn_eff

#     # ---- Option 1: output starts the day AFTER ripening overlap ends ----
#     t_ripe_end = _end_of_longest_true_run_day(daily_ripe)
#     if pd.isna(t_ripe_end):
#         t_out_day = pd.NaT
#     else:
#         t_out_day = (pd.Timestamp(t_ripe_end) + pd.Timedelta(days=1)).normalize()

#     # --- assign daily phase labels ---
#     daily_phase = pd.Series("dry", index=daily, dtype="object")
#     daily_phase.loc[daily_moist] = "moistening"
#     daily_phase.loc[daily_ripe]  = "ripening"
#     if not pd.isna(t_out_day):
#         daily_phase.loc[daily_phase.index >= t_out_day] = "output"

#     # --- map daily -> union acquisition times by day ---
#     phase_u = daily_phase.reindex(t_u.normalize()).values

#     name_to_code = {"dry": 0, "moistening": 1, "ripening": 2, "output": 3}
#     phase_code = np.array([name_to_code.get(p, 0) for p in phase_u], dtype="i4")

#     phase_name = xr.DataArray(
#         phase_u.astype("U10"),
#         coords={"Time": time_u},
#         dims=("Time",),
#         name="phase_name",
#     )

#     phase_code_da = xr.DataArray(
#         phase_code,
#         coords={"Time": time_u},
#         dims=("Time",),
#         name="phase_code",
#         attrs={str(k): v for k, v in PHASE_CODES.items()},
#     )

#     t_out = np.datetime64("NaT", "ns") if pd.isna(t_out_day) else np.datetime64(t_out_day.to_datetime64(), "ns")

#     return xr.Dataset(
#         {
#             "phase_code": phase_code_da,
#             "phase_name": phase_name,
#             "aft_wet": aft_wet_u.rename("aft_wet"),
#             "morn_wet": morn_wet_u.rename("morn_wet"),
#             "output_start_time": xr.DataArray(t_out, name="output_start_time"),
#         }
#     )


def _first_day(series_bool: pd.Series):
    """First index where series is True; else NaT."""
    if series_bool is None or len(series_bool) == 0:
        return pd.NaT
    s = series_bool[series_bool.astype(bool)]
    return s.index[0] if len(s) else pd.NaT

def _first_sustained_false_after_start(series_bool: pd.Series, start_day, sustain_days: int):
    """
    Return first day >= start_day where series_bool is False for sustain_days consecutive days.
    series_bool must be daily-indexed (freq='D') and boolean.
    """
    if pd.isna(start_day):
        return pd.NaT
    s = series_bool.astype(bool).copy()
    s = s.loc[pd.Timestamp(start_day):]  # include start day
    if len(s) == 0:
        return pd.NaT

    false_run = (~s).astype(int)
    # consecutive false streak length
    streak = false_run.groupby((false_run != false_run.shift()).cumsum()).cumsum()
    hits = streak >= sustain_days
    if not hits.any():
        return pd.NaT

    # first day where streak reaches sustain_days
    return hits[hits].index[0]

import numpy as np
import xarray as xr

def classify_sentinel_melt_phases_weighted(
    melt_da: xr.DataArray,
    morning_state: str = "descending",
    afternoon_state: str = "ascending",
    t_min=None,  # <-- Marin output onset timestamp (datetime-like) or None
):
    """
    Hybrid (Recommended):
      - wet/dry from weighted Rc threshold (already in melt_da)
      - moistening: PM wet, AM dry
      - ripening: PM wet, AM wet
      - output: starts at Marin VV-minimum date (t_min), and persists thereafter

    melt_da dims: ('sat:relative_orbit', 'Time')
      values: 1=wet, 0=dry, NaN=no overpass
      coord 'sat:orbit_state' on orbit dim (e.g. ascending/descending)
    """

    PHASE_CODES = {0: "dry", 1: "moistening", 2: "ripening", 3: "output"}

    # ---- masks for AM/PM orbits ----
    aft_mask = melt_da["sat:orbit_state"] == afternoon_state
    morn_mask = melt_da["sat:orbit_state"] == morning_state

    aft_da = melt_da.where(aft_mask, drop=True)
    morn_da = melt_da.where(morn_mask, drop=True)

    # Union acquisition grid is the melt_da "Time" coordinate
    time = melt_da["Time"].astype("datetime64[ns]")
    n = time.size

    # For each union time, "wet" for that state means ANY orbit in that state is wet
    aft_wet = (aft_da == 1).any("sat:relative_orbit").reindex(Time=time, fill_value=False).astype(bool)
    morn_wet = (morn_da == 1).any("sat:relative_orbit").reindex(Time=time, fill_value=False).astype(bool)

    # ---- map t_min (Marin output start) to an index on union acquisition grid ----
    t_out_idx = None
    if t_min is not None and not (np.isnat(np.datetime64(t_min, "ns"))):
        t_out64 = np.datetime64(t_min, "ns")
        tv = time.values.astype("datetime64[ns]")
        # first acquisition >= t_out64
        t_out_idx = int(np.searchsorted(tv, t_out64, side="left"))
        if t_out_idx >= len(tv):
            t_out_idx = None

    # ---- state machine ----
    phase_code = np.zeros(n, dtype="i4")
    state = 0  # 0=dry, 1=moistening, 2=ripening, 3=output

    af = aft_wet.values.astype(bool)
    mo = morn_wet.values.astype(bool)

    for i in range(n):
        # OUTPUT override: once we hit Marin output start, remain output
        if t_out_idx is not None and i >= t_out_idx:
            state = 3
            phase_code[i] = 3
            continue

        if state == 0:  # DRY
            if af[i] and not mo[i]:
                state = 1

        elif state == 1:  # MOISTENING
            if mo[i]:
                state = 2
            elif not af[i]:
                state = 0

        elif state == 2:  # RIPENING
            # remain ripening until output override triggers
            pass

        phase_code[i] = state

    phase_name = xr.DataArray(
        np.array([PHASE_CODES[c] for c in phase_code], dtype="U10"),
        coords={"Time": time},
        dims=("Time",),
        name="phase_name",
    )

    phase_code_da = xr.DataArray(
        phase_code,
        coords={"Time": time},
        dims=("Time",),
        name="phase_code",
        attrs={str(k): v for k, v in PHASE_CODES.items()},
    )

    if t_out_idx is not None:
        t_out = np.asarray(time.values)[t_out_idx].astype("datetime64[ns]")
    else:
        t_out = np.datetime64("NaT", "ns")

    return xr.Dataset(
        {
            "phase_code": phase_code_da,
            "phase_name": phase_name,
            "aft_wet": aft_wet.rename("aft_wet"),
            "morn_wet": morn_wet.rename("morn_wet"),
            "output_start_time": xr.DataArray(t_out, name="output_start_time"),
        }
    )

# def classify_sentinel_melt_phases_weighted(
#     melt_da,
#     morning_state="descending",
#     afternoon_state="ascending",
#     t_min=None,
# ):
#     """
#     Classify Sentinel-1 melt phases from a binary melt DataArray.

#     Parameters
#     ----------
#     melt_da : xr.DataArray
#         dims: ('sat:relative_orbit', 'Time')
#         values: 1 = wet/melt, 0 = dry, NaN = no overpass
#         coord 'sat:orbit_state' on orbit dim with values like
#         'ascending' (afternoon) / 'descending' (morning).
#     morning_state : str
#         Label in sat:orbit_state that corresponds to morning orbit(s).
#     afternoon_state : str
#         Label in sat:orbit_state that corresponds to afternoon orbit(s).
#     t_min : int or np.datetime64 or pandas.Timestamp, optional
#         Index or timestamp of the backscatter minimum (start of OUTPUT).
#         If None, we approximate it as the last time any orbit is wet.

#     Returns
#     -------
#     xr.Dataset with variables:
#         - phase_code (Time,) int [0=dry,1=moistening,2=ripening,3=output]
#         - phase_name (Time,) str
#         - aft_wet (Time,) bool
#         - morn_wet (Time,) bool
#     """

#     PHASE_CODES = {0: "dry", 1: "moistening", 2: "ripening", 3: "output"}

#     time = melt_da["Time"]

#     # ---- derive "afternoon wet" and "morning wet" 1D bool series ----
#     aft_mask = melt_da["sat:orbit_state"] == afternoon_state
#     morn_mask = melt_da["sat:orbit_state"] == morning_state

#     aft_da = melt_da.where(aft_mask, drop=True)
#     morn_da = melt_da.where(morn_mask, drop=True)

#     # True if ANY orbit in that state is wet at that time
#     aft_wet = (aft_da == 1).any("sat:relative_orbit")
#     morn_wet = (morn_da == 1).any("sat:relative_orbit")

#     n = time.size
#     phase_code = np.zeros(n, dtype="i4")

#     # ---- determine index of backscatter minimum (output start) ----
#     if t_min is None:
#         # fallback: last time when any orbit is wet
#         any_wet = (aft_wet | morn_wet).values
#         wet_idx = np.where(any_wet)[0]
#         t_min_idx = int(wet_idx[-1]) if wet_idx.size else None
#     else:
#         # allow int index or datetime-like
#         try:
#             t_min64 = np.datetime64(t_min)
#             # if this succeeds and looks like a datetime, use nearest timestamp
#             if np.issubdtype(t_min64.dtype, np.datetime64):
#                 t_min_idx = int(np.argmin(np.abs(time.values - t_min64)))
#             else:
#                 t_min_idx = int(t_min)
#         except Exception:
#             t_min_idx = int(t_min)

#     # guard against weird indices
#     if t_min_idx is not None and (t_min_idx < 0 or t_min_idx >= n):
#         raise ValueError(f"t_min_idx={t_min_idx} is out of bounds for Time (n={n})")

#     # ---- state machine over time ----
#     state = 0  # 0=dry, 1=moistening, 2=ripening, 3=output
#     af = aft_wet.values
#     mo = morn_wet.values

#     for i in range(n):
#         # OUTPUT override: once output starts, it stays output
#         if t_min_idx is not None and i >= t_min_idx:
#             state = 3
#             phase_code[i] = state
#             continue

#         if state == 0:  # DRY
#             # moistening starts when afternoon orbit is wet and morning is still dry
#             if af[i] and not mo[i]:
#                 state = 1

#         elif state == 1:  # MOISTENING
#             # transitions to ripening once morning becomes wet
#             if mo[i]:
#                 state = 2
#             # or back to dry if afternoon dries before morning ever gets wet
#             elif not af[i]:
#                 state = 0

#         elif state == 2:  # RIPENING
#             # stays ripening until output (handled above)
#             pass

#         phase_code[i] = state

#     # ---- build Dataset ----
#     phase_name = xr.DataArray(
#         np.array([PHASE_CODES[c] for c in phase_code], dtype="U10"),
#         coords={"Time": time},
#         dims=("Time",),
#         name="phase_name",
#     )

#     phase_code_da = xr.DataArray(
#         phase_code,
#         coords={"Time": time},
#         dims=("Time",),
#         name="phase_code",
#         attrs={str(k): v for k, v in PHASE_CODES.items()},
#     )

#     return xr.Dataset(
#         {
#             "phase_code": phase_code_da,
#             "phase_name": phase_name,
#             "aft_wet": aft_wet.rename("aft_wet"),
#             "morn_wet": morn_wet.rename("morn_wet"),
#         }
#     )

# ts is an xr.Dataset with vars like Rc, wet, etc. dims: (sat:relative_orbit, Time)
# or however you built it; adjust as needed.

def rc_series_by_orbit(ts: xr.Dataset) -> dict:
    out = {}
    for o in ts["sat:relative_orbit"].values:
        o = int(o)
        # ensure 1D (Time,) series
        rc = ts["Rc"].sel({"sat:relative_orbit": o})
        # make it a pandas Series with datetime index (this tends to match your Marin code expectations)
        out[o] = pd.Series(rc.values, index=pd.to_datetime(rc["Time"].values)).sort_index()
    return out


def build_phase_intervals_weighted(
    path_,
    cues_y,
    cues_x,
    t_min=None,                 # <-- Marin output onset (datetime-like)
    orbits=(144, 64, 137),
    showOutput=False,
):
    """
    Weighted wetness for AM/PM states + Marin output start (t_min).
    """
    da_all = []
    wet_snow_all = []
    melt_time_all = []

    for orbit in orbits:
        wet_snow_tseries, da_tseries, melt_time = weighted_wet_snow_by_orbit(
            path_, cues_y, cues_x, orbit, showOutput=showOutput
        )
        da_all.append(da_tseries)
        wet_snow_all.append(wet_snow_tseries)
        melt_time_all.append(melt_time)

    combined_orbits = xr.concat(
        da_all,
        dim="sat:relative_orbit",
        join="outer",
    )

    ds_phase = classify_sentinel_melt_phases_weighted(
        combined_orbits,
        t_min=t_min,
    )
    return ds_phase


def phases_time_to_intervals_weighted(
    ds_time: xr.Dataset,
    time_dim: str = "Time",
    phase_id_var: str = "phase_code",
    phase_name_var: str = "phase_name",
    out_dim: str = "interval",
    drop_last_open: bool = True,
    time_min: str = None,   # e.g. "2016-10-01"
    time_max: str = None,   # e.g. "2017-09-30"
    dedupe: str = "last",   # "last" | "first" | None
) -> xr.Dataset:
    """
    Convert a time-indexed phase dataset (Time, phase_code/phase_name) into an
    interval-indexed dataset like your ph_s1_vv_*:
        coords: interval_start (interval), interval_end (interval)
        data:   phase_id (interval), phase_name (interval)

    Intervals are defined as [t_i, t_{i+1}) and labeled using phase at t_i.
    """

    # --- checks ---
    if time_dim not in ds_time.dims:
        raise ValueError(f"Expected time dim '{time_dim}' in ds_time.dims={ds_time.dims}")
    if phase_id_var not in ds_time.data_vars:
        raise ValueError(f"Expected '{phase_id_var}' in ds_time.data_vars={list(ds_time.data_vars)}")
    if phase_name_var not in ds_time.data_vars:
        raise ValueError(f"Expected '{phase_name_var}' in ds_time.data_vars={list(ds_time.data_vars)}")

    # --- time index ---
    t = pd.to_datetime(ds_time[time_dim].values)
    if len(t) < 2:
        raise ValueError("Need at least 2 times to form intervals.")

    # sort by time
    order = np.argsort(t)
    t = t[order]
    ds = ds_time.isel({time_dim: order})

    # optional window clip (keep any samples within bounds)
    if time_min is not None:
        tmin = pd.Timestamp(time_min)
        keep = t >= tmin
        t = t[keep]
        ds = ds.isel({time_dim: np.where(keep)[0]})
    if time_max is not None:
        tmax = pd.Timestamp(time_max)
        keep = t <= tmax
        t = t[keep]
        ds = ds.isel({time_dim: np.where(keep)[0]})

    if len(t) < 2:
        raise ValueError("Not enough times after clipping to form intervals.")

    # optional dedupe (handles repeated identical timestamps)
    if dedupe in ("last", "first"):
        # indices of unique times
        # groupby in pandas to choose first/last occurrence
        s = pd.Series(np.arange(len(t)), index=t)
        if dedupe == "last":
            keep_idx = s.groupby(level=0).max().values
        else:
            keep_idx = s.groupby(level=0).min().values
        keep_idx = np.sort(keep_idx)

        t = t[keep_idx]
        ds = ds.isel({time_dim: keep_idx})

    if len(t) < 2:
        raise ValueError("Not enough unique times after dedupe to form intervals.")

    # --- build intervals [t_i, t_{i+1}) ---
    interval_start = t[:-1]
    interval_end = t[1:]

    # label intervals using phase at t_i
    phase_id = ds[phase_id_var].isel({time_dim: slice(0, -1)}).values
    phase_nm = ds[phase_name_var].isel({time_dim: slice(0, -1)}).values

    # optionally drop last interval (usually already dropped because we used t[:-1])
    # but keep this for symmetry if you change logic later
    if not drop_last_open:
        # If you wanted to keep an "open-ended" last interval, you'd need a rule for interval_end.
        # For now we keep the default: intervals only where end exists.
        pass

    # --- output dataset ---
    out = xr.Dataset(
        data_vars={
            "phase_id": xr.DataArray(
                phase_id, dims=[out_dim],
                coords={"interval_start": (out_dim, interval_start)}
            ),
            "phase_name": xr.DataArray(
                phase_nm, dims=[out_dim],
                coords={"interval_start": (out_dim, interval_start)}
            ),
            "interval_end": xr.DataArray(
                pd.to_datetime(interval_end), dims=[out_dim],
                coords={"interval_start": (out_dim, interval_start)}
            ),
        }
    ).set_coords(["interval_end"])

    return out








def build_phase_intervals_vv(
    wet_pm: xr.DataArray,            # afternoon wet flag (e.g. orbit 137)
    wet_am: Optional[xr.DataArray],    # morning wet flag (e.g. orbit 144) - can be None for now
    output_start: pd.Timestamp,      # mean minima time across orbits
    allow_ripe_end_two_dry: bool = True,
) -> xr.Dataset:
    """
    Returns a Dataset with interval_start times and phase_id + phase_name.
    Phases apply to [t_i, t_{i+1}) intervals on the union acquisition grid.
    """

    wet_pm = _to_1d_bool(wet_pm)
    times = pd.to_datetime(wet_pm["time"].values)

    if wet_am is not None:
        wet_am = _to_1d_bool(wet_am)
        times = times.union(pd.to_datetime(wet_am["time"].values))

    times = pd.DatetimeIndex(times).sort_values()
    if len(times) < 2:
        raise ValueError("Need at least 2 unique acquisition times to form intervals.")

    # Reindex to union time grid and forward-fill each series independently
    wet_pm_u = wet_pm.reindex(time=times).ffill("time")
    wet_am_u = None
    if wet_am is not None:
        wet_am_u = wet_am.reindex(time=times).ffill("time")

    # We’ll label intervals [t_i, t_{i+1}) using wet state at t_i
    t0 = times[:-1]
    t1 = times[1:]

    # State machine flags
    in_ripe = False
    consecutive_am_dry = 0

    phase_id = np.zeros(len(t0), dtype=np.int16)  # default dry=0

    for i, ti in enumerate(t0):
        # Rule 1: output starts at output_start (use interval start)
        if pd.Timestamp(ti) >= output_start:
            phase_id[i] = 3  # output
            continue

        pm_wet = bool(wet_pm_u.sel(time=ti).values)

        am_wet = False
        if wet_am_u is not None:
            am_wet = bool(wet_am_u.sel(time=ti).values)

        # Rule 2: moist = afternoon wet but morning not
        if (wet_am_u is not None) and pm_wet and (not am_wet) and (not in_ripe):
            phase_id[i] = 1  # moist
            continue

        # Rule 3: ripe starts once morning becomes wet
        if wet_am_u is not None:
            if (not in_ripe) and am_wet:
                in_ripe = True
                consecutive_am_dry = 0

            if in_ripe:
                # keep ripe until output OR two consecutive morning-dry retrievals
                if am_wet:
                    consecutive_am_dry = 0
                else:
                    consecutive_am_dry += 1

                if allow_ripe_end_two_dry and (consecutive_am_dry >= 2):
                    in_ripe = False
                    phase_id[i] = 0  # back to dry (or you could choose moist if pm_wet)
                else:
                    phase_id[i] = 2  # ripe
                continue

        # Otherwise dry (0)
        phase_id[i] = 0

    ds_out = xr.Dataset(
        {
            "phase_id": xr.DataArray(phase_id, dims=["interval"], coords={"interval_start": ("interval", t0)}),
            "interval_end": xr.DataArray(pd.to_datetime(t1), dims=["interval"], coords={"interval_start": ("interval", t0)}),
        }
    )

    ds_out["phase_name"] = PHASE.sel(phase_id=ds_out["phase_id"]).rename("phase_name")
    ds_out = ds_out.set_coords(["interval_end"])
    return ds_out

def _to_1d(da):
    """
    Normalize a 1D time series DataArray.

    Accepts both 'time' and 'Time' dim names (we have both in this repo).
    Returns a DataArray with dim named 'time'.
    """
    import xarray as xr

    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(da)}")

    # --- normalize time dimension name ---
    if "time" not in da.dims and "Time" in da.dims:
        da = da.rename({"Time": "time"})

    if "time" not in da.dims:
        raise ValueError(f"Expected a DataArray with a 'time' dimension. Got dims={da.dims}")

    # squeeze any singleton dims like band, sat:relative_orbit
    da1 = da.squeeze(drop=True)

    # ensure 1D
    if da1.ndim != 1 or da1.dims != ("time",):
        # if it’s still not 1D, this makes the error message actionable
        raise ValueError(f"Expected 1D time series after squeeze. dims={da1.dims}, shape={da1.shape}")

    return da1

def output_start_mean_min_vv_window(
    vv_by_orbit: dict,
    year: int,
    start_md: str = "02-01",
    end_md: str = "08-01",
) -> pd.Timestamp:
    """
    Mean of per-orbit *raw VV* minima within [Feb 1, Aug 1] of the given year.
    vv_by_orbit: dict like {"pm_137": vv_da, "am_144": vv_da, ...}
    """
    start = f"{year}-{start_md}"
    end   = f"{year}-{end_md}"

    mins = []
    for k, vv in vv_by_orbit.items():
        vv1 = _to_1d(vv).sel(time=slice(start, end))
        if vv1.sizes.get("time", 0) == 0:
            raise ValueError(f"No data for {k} in window {start} to {end}.")
        mins.append(pd.Timestamp(vv1.idxmin("time").item()))

    mean_ns = int(np.mean([m.value for m in mins]))
    return pd.to_datetime(mean_ns)


def output_start_mean_min_dvv_window(
    vv_by_orbit: dict,
    year: int,
    baseline_md=("01-01", "02-01"),
    search_md=("02-01", "08-01"),
) -> pd.Timestamp:
    """
    Mean of per-orbit minima of ΔVV = VV - median(VV in baseline window),
    searched within [Feb 1, Aug 1].

    This is more robust than raw VV because it anchors to winter conditions.
    """
    b0 = f"{year}-{baseline_md[0]}"
    b1 = f"{year}-{baseline_md[1]}"
    s0 = f"{year}-{search_md[0]}"
    s1 = f"{year}-{search_md[1]}"

    mins = []
    for k, vv in vv_by_orbit.items():
        vv1 = _to_1d(vv)

        base = vv1.sel(time=slice(b0, b1))
        if base.sizes.get("time", 0) == 0:
            raise ValueError(f"No baseline data for {k} in window {b0} to {b1}.")
        ref = base.median("time")

        dvv = (vv1 - ref).sel(time=slice(s0, s1))
        if dvv.sizes.get("time", 0) == 0:
            raise ValueError(f"No search data for {k} in window {s0} to {s1}.")
        mins.append(pd.Timestamp(dvv.idxmin("time").item()))

    mean_ns = int(np.mean([m.value for m in mins]))
    return pd.to_datetime(mean_ns)

def output_start_joint_localmin_dvv(
    vv_by_orbit: dict,
    year: int,
    baseline_md=("01-01", "02-01"),
    search_md=("02-01", "08-01"),
    anchor_orbit="137",          # try to anchor to this orbit key if it exists
    window_days=20,              # match other orbit minima within +/- this many days of anchor
    smooth_k=3,                  # rolling median on acquisition index (not daily)
    future_k=3,                  # lookahead acquisitions for "recovery"
    min_recovery_db=0.8,         # require dvv to rise by this much after the minimum
    reducer="mean",              # "mean" or "median" of matched orbit dates
    debug=False,
) -> pd.Timestamp:
    """
    Improved output-start estimate:
      - compute ΔVV relative to winter baseline (same as your function)
      - select a "turning-point" local minimum (not necessarily absolute minimum)
      - enforce cross-orbit consistency by matching minima within a time window

    Returns a single Timestamp.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    def _rolling_median_1d(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1 or arr.size < k:
            return arr.copy()
        out = arr.copy()
        half = k // 2
        for i in range(arr.size):
            i0 = max(0, i - half)
            i1 = min(arr.size, i + half + 1)
            out[i] = np.nanmedian(arr[i0:i1])
        return out

    def _local_minima_idx(y: np.ndarray) -> np.ndarray:
        # indices i where y[i] is a local minimum (handles plateaus conservatively)
        if y.size < 3:
            return np.array([], dtype=int)
        left = y[1:-1] <= y[:-2]
        right = y[1:-1] <= y[2:]
        return np.where(left & right)[0] + 1

    def _pick_best_minimum(t: pd.DatetimeIndex, y: np.ndarray) -> pd.Timestamp:
        """
        Pick a plausible local minimum:
          - must be followed by recovery >= min_recovery_db within next future_k acquisitions
          - among candidates, choose the one with max recovery (tie-break by deepest y)
        Falls back to absolute min if no candidate.
        """
        idxs = _local_minima_idx(y)
        candidates = []
        for i in idxs:
            # need enough points ahead to evaluate recovery
            j1 = min(y.size, i + 1 + future_k)
            if j1 <= i + 1:
                continue
            future_max = np.nanmax(y[i+1:j1])
            recovery = future_max - y[i]
            if np.isfinite(recovery) and recovery >= min_recovery_db:
                candidates.append((recovery, -y[i], i))  # maximize recovery, then minimize y

        if candidates:
            candidates.sort(reverse=True)
            best_i = candidates[0][2]
            return pd.Timestamp(t[best_i])

        # fallback: absolute minimum
        return pd.Timestamp(t[int(np.nanargmin(y))])

    b0 = f"{year}-{baseline_md[0]}"
    b1 = f"{year}-{baseline_md[1]}"
    s0 = f"{year}-{search_md[0]}"
    s1 = f"{year}-{search_md[1]}"

    per_orbit = {}  # orbit -> dict with dvv series + chosen candidates
    for k, vv in vv_by_orbit.items():
        vv1 = _to_1d(vv)

        base = vv1.sel(time=slice(b0, b1))
        if base.sizes.get("time", 0) == 0:
            raise ValueError(f"No baseline data for {k} in window {b0} to {b1}.")
        ref = base.median("time")

        dvv = (vv1 - ref).sel(time=slice(s0, s1))
        if dvv.sizes.get("time", 0) == 0:
            raise ValueError(f"No search data for {k} in window {s0} to {s1}.")

        t = pd.to_datetime(dvv["time"].values)
        y = dvv.values.astype(float)

        y_s = _rolling_median_1d(y, smooth_k)
        chosen = _pick_best_minimum(pd.DatetimeIndex(t), y_s)
        # chosen = _pick_output_min_with_persistence(pd.DatetimeIndex(t), y_s)

        per_orbit[k] = dict(times=pd.DatetimeIndex(t), dvv=y, dvv_s=y_s, chosen=chosen)

    # --- choose anchor date ---
    anchor_key = None
    # try exact match
    if anchor_orbit in per_orbit:
        anchor_key = anchor_orbit
    else:
        # try substring match (handles keys like "orbit_137" or "S1_137")
        for k in per_orbit:
            if anchor_orbit in str(k):
                anchor_key = k
                break

    if anchor_key is None:
        # no anchor found: just aggregate chosen across all orbits
        chosen_dates = [v["chosen"] for v in per_orbit.values()]
        if reducer == "median":
            ns = np.median([d.value for d in chosen_dates])
        else:
            ns = np.mean([d.value for d in chosen_dates])
        out = pd.to_datetime(int(ns))
        if debug:
            print("No anchor orbit found; using aggregate of per-orbit chosen minima:", chosen_dates, "->", out)
        return out

    anchor_date = per_orbit[anchor_key]["chosen"]

    # --- match other orbits to anchor within window_days ---
    matched = {}
    for k, info in per_orbit.items():
        t = info["times"]
        y_s = info["dvv_s"]

        # all plausible local mins for matching (using same recovery filter as selector)
        idxs = _local_minima_idx(y_s)
        candidates = []
        for i in idxs:
            j1 = min(y_s.size, i + 1 + future_k)
            if j1 <= i + 1:
                continue
            future_max = np.nanmax(y_s[i+1:j1])
            recovery = future_max - y_s[i]
            if np.isfinite(recovery) and recovery >= min_recovery_db:
                dt = pd.Timestamp(t[i])
                # distance to anchor (days)
                dist = abs((dt - anchor_date).days)
                if dist <= window_days:
                    # prefer closer to anchor, then stronger recovery, then deeper min
                    candidates.append(( -dist, recovery, -y_s[i], dt))

        if candidates:
            candidates.sort(reverse=True)
            matched[k] = candidates[0][3]
        else:
            # fallback to chosen (which might be absolute min)
            matched[k] = info["chosen"]

    chosen_dates = list(matched.values())
    if reducer == "median":
        ns = np.median([d.value for d in chosen_dates])
    else:
        ns = np.mean([d.value for d in chosen_dates])

    out = pd.to_datetime(int(ns))

    if debug:
        print(f"Anchor orbit: {anchor_key} -> {anchor_date}")
        for k in matched:
            print(f"  {k}: matched {matched[k]} (raw chosen {per_orbit[k]['chosen']})")
        print("Final output_start:", out)

    return out

def _pick_output_min_with_persistence(
    t: pd.DatetimeIndex,
    y_s: np.ndarray,
    future_k: int = 4,
    min_recovery_allowed_db: float = 0.5,   # max allowed rebound above the min (small)
    frac_below_min_plus: float = 0.75,      # fraction of next points that must remain near/below min+delta
    near_min_delta_db: float = 1.0,         # “near-min” band
):
    """
    Pick earliest local minimum that looks like the start of a sustained low regime:
      - look ahead up to future_k acquisitions
      - require most of those points remain <= (min + near_min_delta_db)
      - and do NOT rebound strongly above (min + min_recovery_allowed_db + near_min_delta_db)
    Fallback: absolute minimum.
    """
    idxs = np.array([], dtype=int)
    if y_s.size >= 3:
        left = y_s[1:-1] <= y_s[:-2]
        right = y_s[1:-1] <= y_s[2:]
        idxs = np.where(left & right)[0] + 1

    def ok_candidate(i: int) -> bool:
        j1 = min(y_s.size, i + 1 + future_k)
        if j1 <= i + 2:
            return False
        future = y_s[i+1:j1]
        minv = y_s[i]
        near = minv + near_min_delta_db
        frac = np.mean(future <= near)
        if frac < frac_below_min_plus:
            return False
        # prevent strong rebound
        if np.nanmax(future) > (near + min_recovery_allowed_db):
            return False
        return True

    for i in idxs:
        if ok_candidate(i):
            return pd.Timestamp(t[i])

    # fallback
    return pd.Timestamp(t[int(np.nanargmin(y_s))])


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

def _ref_to_scalar(ref_obj, orbit_key: str = None) -> float:
    """
    Convert various 'ref' objects into a scalar float.
    Supports:
      - float/int
      - numpy scalar
      - pandas Series with one value
      - xarray DataArray/Dataset with one value
    If Dataset has multiple vars, uses `orbit_key` if provided, otherwise takes first variable.
    """
    # plain numeric
    if isinstance(ref_obj, (float, int, np.floating, np.integer)):
        return float(ref_obj)

    # pandas
    if hasattr(ref_obj, "values") and str(type(ref_obj)).endswith("pandas.core.series.Series'>"):
        vals = np.asarray(ref_obj.values).ravel()
        if vals.size == 1:
            return float(vals[0])

    # xarray DataArray
    try:
        import xarray as xr
        if isinstance(ref_obj, xr.DataArray):
            vals = np.asarray(ref_obj.values).ravel()
            if vals.size == 1:
                return float(vals[0])
            # if it's not size 1, user likely forgot to index; raise
            raise ValueError("ref DataArray is not scalar; index it to a single point first.")
        if isinstance(ref_obj, xr.Dataset):
            # choose variable
            if orbit_key is not None and orbit_key in ref_obj.data_vars:
                da = ref_obj[orbit_key]
            else:
                # take first var
                v0 = list(ref_obj.data_vars)[0]
                da = ref_obj[v0]
            vals = np.asarray(da.values).ravel()
            if vals.size == 1:
                return float(vals[0])
            raise ValueError("ref Dataset variable is not scalar; index it to a single point first.")
    except Exception:
        pass

    raise TypeError(f"Unsupported ref type for scalar conversion: {type(ref_obj)}")


def _bsc_to_series(bsc_obj) -> pd.Series:
    """
    Convert backscatter container into a pandas Series indexed by datetime.
    Supports:
      - pandas Series (returns as-is)
      - xarray DataArray with a 'time' dimension
    """
    if isinstance(bsc_obj, pd.Series):
        s = bsc_obj.copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    try:
        import xarray as xr
        if isinstance(bsc_obj, xr.DataArray):
            # ensure time dimension exists
            if "time" not in bsc_obj.dims:
                raise ValueError(f"Expected 'time' dim in DataArray, got dims={bsc_obj.dims}")

            # If it's more than 1D (e.g., has band/orbit), squeeze it down
            da = bsc_obj
            # drop/squeeze non-time dims if they're length-1
            for d in list(da.dims):
                if d != "time" and da.sizes.get(d, 1) == 1:
                    da = da.squeeze(d, drop=True)

            # After squeezing, we expect 1D time series
            if da.ndim != 1:
                raise ValueError(f"DataArray is not 1D after squeeze; dims={da.dims}, shape={da.shape}")

            s = da.to_series()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    except Exception:
        pass

    raise TypeError(f"Unsupported bsc type for series conversion: {type(bsc_obj)}")


@dataclass
class MarinS1VVCfg:
    tau_db: float = -3.0                    # threshold on ΔVV (VV - ref)
    frac_below_bad: float = 0.6             # BAD_REFERENCE_OR_NO_CONTRAST if >= this
    max_am_pm_mismatch_days: int = 28       # AM vs PM minima mismatch tolerance (≈ 1 pass)
    max_group_spread_days: int = 28         # if multiple AM or PM orbits, require internal agreement
    local_window_days: int = 30             # local window around min for distinctness/persistence
    min_stability_days: int = 28            # jackknife spread threshold for stability
    min_distinct_db: float = 0.7            # q10(local) - min must exceed this
    persist_margin_db: float = 0.5          # local points <= (min + margin) count as supporting lows
    persist_min_count: int = 2              # require min itself + >=1 supporting point


def _median_date(dates: List[pd.Timestamp]) -> pd.Timestamp:
    dt = pd.to_datetime(dates)
    as_int = dt.view("i8")
    return pd.to_datetime(int(np.median(as_int)))

def _days_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return abs((a - b).total_seconds()) / 86400.0

def _slice_year_window(series: pd.Series, year: int, search_md: Tuple[str, str]) -> pd.Series:
    t0 = pd.Timestamp(f"{year}-{search_md[0]}")
    t1 = pd.Timestamp(f"{year}-{search_md[1]}")
    return series.loc[(series.index >= t0) & (series.index <= t1)].dropna()

def _tmin_amin(delta: pd.Series) -> Tuple[pd.Timestamp, float]:
    idx = pd.Timestamp(delta.idxmin())
    return idx, float(delta.loc[idx])

def _jackknife_spread_days(delta: pd.Series) -> Optional[float]:
    delta = delta.dropna()
    if delta.shape[0] < 3:
        return None
    mins = []
    for i in range(delta.shape[0]):
        d2 = delta.drop(delta.index[i])
        if d2.empty:
            continue
        mins.append(pd.Timestamp(d2.idxmin()))
    if len(mins) < 2:
        return None
    return float((max(mins) - min(mins)).days)

def _noisy_signal(delta: pd.Series, t_min: pd.Timestamp, a_min: float, cfg: MarinS1VVCfg) -> Tuple[bool, Dict[str, Any]]:
    diag: Dict[str, Any] = {}

    # stability of minima date
    jk = _jackknife_spread_days(delta)
    diag["jackknife_spread_days"] = jk
    stability_fail = (jk is not None) and (jk > cfg.min_stability_days)

    # local neighborhood around minima
    tw0 = t_min - pd.Timedelta(days=cfg.local_window_days)
    tw1 = t_min + pd.Timedelta(days=cfg.local_window_days)
    local = delta.loc[(delta.index >= tw0) & (delta.index <= tw1)].dropna()
    diag["local_n"] = int(local.shape[0])

    # distinctness: q10(local) - min
    if local.shape[0] >= 3:
        q10 = float(local.quantile(0.10))
        distinct = q10 - a_min
        diag["distinct_q10_minus_min_db"] = distinct
        distinct_fail = distinct < cfg.min_distinct_db
    else:
        diag["distinct_q10_minus_min_db"] = None
        distinct_fail = True

    # persistence: need min + at least one other low in local window
    low = local <= (a_min + cfg.persist_margin_db)
    low_count = int(low.sum())
    diag["local_low_count"] = low_count
    persist_fail = low_count < cfg.persist_min_count

    noisy = stability_fail and (distinct_fail or persist_fail)
    diag["stability_fail"] = stability_fail
    diag["distinct_fail"] = distinct_fail
    diag["persist_fail"] = persist_fail
    diag["noisy"] = noisy
    return noisy, diag


def output_start_marin_minima_qc(
    bsc_by_orbit: Dict[str, pd.Series],
    ref_by_orbit: Dict[str, float],
    orbit_group: Dict[str, str],            # orbit -> "morning" or "afternoon"
    year: int,
    search_md: Tuple[str, str] = ("01-01", "07-01"),
    cfg: MarinS1VVCfg = MarinS1VVCfg(),
) -> Dict[str, Any]:
    """
    Marin-style "output start" = mean( median AM minima date, median PM minima date )
    with QC (returns pd.NaT for t_out when failing).

    Expects:
      - bsc_by_orbit: orbit -> backscatter Series (dB) indexed by datetime
      - ref_by_orbit: orbit -> scalar reference (dB)
      - orbit_group: orbit -> "morning" or "afternoon"

    Returns dict:
      - output_start (pd.Timestamp or pd.NaT)
      - qc_ok (bool)
      - qc_reason (str)
      - per_orbit diagnostics + group summaries
    """

    per: Dict[str, Any] = {}
    good_morn: List[str] = []
    good_aft: List[str] = []

    for o, bsc in bsc_by_orbit.items():
        if bsc is None:
            continue
        if o not in ref_by_orbit or ref_by_orbit[o] is None:
            per[o] = {"ok": False, "reason": "NO_REFERENCE", "t_min": None, "a_min": None}
            continue

        # ref = float(ref_by_orbit[o])
        ref = _ref_to_scalar(ref_by_orbit[o], orbit_key=o)
        # bsc = bsc.sort_index()
        bsc = _bsc_to_series(bsc)

        # slice to search window (this is where minima is defined)
        bsc_win = _slice_year_window(bsc, year, search_md)
        if bsc_win.empty:
            per[o] = {"ok": False, "reason": "NO_DATA_IN_SEARCH_WINDOW", "ref": ref}
            continue

        delta = bsc_win - ref
        t_min, a_min = _tmin_amin(delta)

        f_below = float((delta < cfg.tau_db).mean())

        # contrast checks (these are your "bad reference" and "no trigger" modes)
        if f_below >= cfg.frac_below_bad:
            per[o] = {"ok": False, "reason": "BAD_REFERENCE_OR_NO_CONTRAST", "ref": ref,
                      "t_min": t_min, "a_min": a_min, "f_below": f_below}
            continue

        if f_below == 0.0:
            per[o] = {"ok": False, "reason": "NO_THRESHOLD_CROSSING", "ref": ref,
                      "t_min": t_min, "a_min": a_min, "f_below": f_below}
            continue

        # tightened noisy test (minima-centered)
        noisy, noisy_diag = _noisy_signal(delta, t_min, a_min, cfg)
        if noisy:
            per[o] = {"ok": False, "reason": "NOISY_SIGNAL", "ref": ref,
                      "t_min": t_min, "a_min": a_min, "f_below": f_below,
                      "noisy_diag": noisy_diag}
            continue

        per[o] = {"ok": True, "reason": "OK", "ref": ref,
                  "t_min": t_min, "a_min": a_min, "f_below": f_below,
                  "noisy_diag": noisy_diag}

        grp = orbit_group.get(o)
        if grp == "morning":
            good_morn.append(o)
        elif grp == "afternoon":
            good_aft.append(o)

    # Marin requirement: at least one GOOD morning and one GOOD afternoon
    if len(good_morn) == 0:
        return {"output_start": pd.NaT, "qc_ok": False, "qc_reason": "NO_GOOD_MORNING_ORBIT", "per_orbit": per}
    if len(good_aft) == 0:
        return {"output_start": pd.NaT, "qc_ok": False, "qc_reason": "NO_GOOD_AFTERNOON_ORBIT", "per_orbit": per}

    # group medians
    t_morn = _median_date([per[o]["t_min"] for o in good_morn])
    t_aft  = _median_date([per[o]["t_min"] for o in good_aft])

    # within-group spread if multiple good orbits
    def _spread_days(orbits: List[str]) -> float:
        if len(orbits) < 2:
            return 0.0
        ds = [per[o]["t_min"] for o in orbits]
        return float((max(ds) - min(ds)).days)

    morn_spread = _spread_days(good_morn)
    aft_spread  = _spread_days(good_aft)

    if morn_spread > cfg.max_group_spread_days:
        return {"output_start": pd.NaT, "qc_ok": False, "qc_reason": "INCONSISTENT_MORNING_MINIMA",
                "per_orbit": per, "t_morning": t_morn, "t_afternoon": t_aft,
                "morning_spread_days": morn_spread, "afternoon_spread_days": aft_spread,
                "good_morning_orbits": good_morn, "good_afternoon_orbits": good_aft}

    if aft_spread > cfg.max_group_spread_days:
        return {"output_start": pd.NaT, "qc_ok": False, "qc_reason": "INCONSISTENT_AFTERNOON_MINIMA",
                "per_orbit": per, "t_morning": t_morn, "t_afternoon": t_aft,
                "morning_spread_days": morn_spread, "afternoon_spread_days": aft_spread,
                "good_morning_orbits": good_morn, "good_afternoon_orbits": good_aft}

    # AM/PM mismatch
    if _days_between(t_morn, t_aft) > cfg.max_am_pm_mismatch_days:
        return {"output_start": pd.NaT, "qc_ok": False, "qc_reason": "AM_PM_MINIMA_MISMATCH",
                "per_orbit": per, "t_morning": t_morn, "t_afternoon": t_aft,
                "morning_spread_days": morn_spread, "afternoon_spread_days": aft_spread,
                "good_morning_orbits": good_morn, "good_afternoon_orbits": good_aft}

    # Marin output: mean of the two group dates
    output_start = pd.to_datetime(int(np.mean([t_morn.value, t_aft.value])))

    return {"output_start": output_start, "qc_ok": True, "qc_reason": "OK",
            "per_orbit": per, "t_morning": t_morn, "t_afternoon": t_aft,
            "morning_spread_days": morn_spread, "afternoon_spread_days": aft_spread,
            "good_morning_orbits": good_morn, "good_afternoon_orbits": good_aft}


def run_s1_vv_year(
    sentinel_year,
    sentinel_ref_year,
    idy,
    idx,
    year,
    cfg,
    search_md=("01-01", "07-01"),
):
    wet64, wet137, wet144, bsc64, bsc137, bsc144, ref64, ref137, ref144 = \
        s1_bsc_ref(
            sentinel_year,
            sentinel_ref_year,
            idy_=idy,
            idx_=idx,
            showPlot=False,
        )

    res = output_start_marin_minima_qc(
        bsc_by_orbit={"64": bsc64, "137": bsc137, "144": bsc144},
        ref_by_orbit={"64": ref64, "137": ref137, "144": ref144},
        orbit_group={"64": "afternoon", "137": "afternoon", "144": "morning"},
        year=year,
        search_md=search_md,
        cfg=cfg,
    )

    # Merge ascending (PM) orbits 64 and 137, but only include orbits that passed QC.
    # Wet flags from QC-failed orbits are unreliable (e.g. persistently wet-looking).
    # wet64/wet137 may have singleton spatial dims (time, y=1, x=1) — squeeze to 1D first.
    per = res.get("per_orbit", {})
    wet64_1d  = wet64.squeeze(drop=True).astype(bool)
    wet137_1d = wet137.squeeze(drop=True).astype(bool)
    pm_wet_series = []
    if per.get("64", {}).get("ok", False):
        pm_wet_series.append(wet64_1d.to_series())
    if per.get("137", {}).get("ok", False):
        pm_wet_series.append(wet137_1d.to_series())
    if pm_wet_series:
        wet_pm_s = pd.concat(pm_wet_series, axis=1).any(axis=1)
        wet_pm = xr.DataArray(
            wet_pm_s.values.astype(bool),
            dims=wet137_1d.dims,
            coords={wet137_1d.dims[0]: wet_pm_s.index},
        )
    else:
        # No PM orbit passed QC → phase builder will return NaT
        wet_pm = xr.DataArray(
            np.array([], dtype=bool), dims=["time"], coords={"time": []}
        )

    phases = build_phase_intervals_ripe_start_both_stop_either_dry(
        wet_pm=wet_pm,
        wet_am=wet144,
        output_start=res["output_start"],
        allow_ripe_end_two_dry=True,
    )

    return {
        "output_start": res["output_start"],
        "qc_reason": res["qc_reason"],
        "qc_ok": res["qc_ok"],
        "phases": phases,
        "diagnostics": res,
    }



# --- reuse your existing cfg dataclass ---
# MarinS1VVCfg is fine; we'll just set tau_db appropriate for Rc deltas.

def _rc_to_series(da_1d: xr.DataArray) -> pd.Series:
    """Convert 1D xarray DataArray (Time,) -> pandas Series indexed by datetime."""
    t = pd.to_datetime(da_1d["Time"].values)
    s = pd.Series(da_1d.values.astype(float), index=t).sort_index()
    return s.dropna()


def _baseline_ref(series: pd.Series, year: int, baseline_md=("01-01", "02-01")) -> float:
    """Median in baseline window (same idea as VV)."""
    b0 = pd.Timestamp(f"{year}-{baseline_md[0]}")
    b1 = pd.Timestamp(f"{year}-{baseline_md[1]}")
    base = series.loc[(series.index >= b0) & (series.index <= b1)].dropna()
    if base.empty:
        raise ValueError(f"No baseline data in {b0}..{b1}")
    return float(base.median())


def run_s1_rc_year_from_ts(
    ts: xr.Dataset,
    year: int,
    cfg,
    *,
    baseline_md=("01-01", "02-01"),
    search_md=("01-01", "07-01"),
    rc_var="Rc",
    wet_var="wet",
    orbit_dim="sat:relative_orbit",
    state_coord="sat:orbit_state",
):
    """
    Marin-style minima+QC, but using Rc (weighted VV/VH metric) instead of VV.
    Expects ts to have:
      - ts[rc_var] with dims (sat:relative_orbit, Time)  [or selectable to (Time,) per orbit]
      - ts[wet_var] with dims (sat:relative_orbit, Time) boolean or 0/1
      - coord ts[state_coord] on orbit_dim giving "ascending"/"descending"
    Returns the same dict structure as run_s1_vv_year.
    """

    if orbit_dim not in ts.dims:
        raise ValueError(f"Expected orbit dim {orbit_dim!r} in ts.dims={ts.dims}")
    if "Time" not in ts.dims:
        raise ValueError(f"Expected 'Time' in ts.dims={ts.dims}")
    if rc_var not in ts:
        raise KeyError(f"ts missing {rc_var!r}. Vars: {list(ts.data_vars)}")
    if wet_var not in ts:
        raise KeyError(f"ts missing {wet_var!r}. Vars: {list(ts.data_vars)}")
    if state_coord not in ts.coords:
        raise KeyError(f"ts missing coord {state_coord!r}. Coords: {list(ts.coords)}")

    # ---- build per-orbit Rc series + baseline refs ----
    bsc_by_orbit = {}
    ref_by_orbit = {}
    orbit_group = {}

    for o in ts[orbit_dim].values:
        o_int = str(int(o))

        rc_o = ts[rc_var].sel({orbit_dim: o})
        # squeeze to 1D (Time,)
        rc_o = rc_o.squeeze()

        s_rc = _rc_to_series(rc_o)
        if s_rc.empty:
            continue

        bsc_by_orbit[o_int] = s_rc
        
        # CRITICAL: For Weighted Rc, the reference cosmetic correction is ALREADY
        # built into the Rc formula per Nagler et al. (2016):
        #   Rc = W(θ) * ΔVH + (1-W(θ)) * ΔVV
        # where ΔVV and ΔVH are already relative to their baselines.
        # Therefore, threshold should apply directly to Rc, NOT to (Rc - reference).
        # Set ref = 0.0 so delta = Rc - 0 = Rc, then check delta < tau_db works correctly.
        ref_by_orbit[o_int] = 0.0

        st = str(ts[state_coord].sel({orbit_dim: o}).values)
        # map to Marin-style group labels
        orbit_group[o_int] = "afternoon" if st == "ascending" else "morning"

    # ---- run Marin QC/minima on Rc ----
    res = output_start_marin_minima_qc(
        bsc_by_orbit=bsc_by_orbit,
        ref_by_orbit=ref_by_orbit,
        orbit_group=orbit_group,
        year=year,
        search_md=search_md,
        cfg=cfg,
    )

    # ---- Build wet_pm / wet_am from ts[wet] (so your phase builder can run) ----
    # For CUES: PM = any ascending orbit wet; AM = any descending orbit wet.
    # CRITICAL FIX: Only include times where orbits ACTUALLY observed
    # The merged ts has dims (sat:relative_orbit, Time) with outer join, so many slots are NaN
    # We must extract only times/values where each orbit type actually has data
    # Only include orbits that passed QC — QC-failed orbits have unreliable wet flags.

    per = res.get("per_orbit", {})
    wet_pm_list = []
    wet_am_list = []

    for o in ts[orbit_dim].values:
        o_int = str(int(o))
        if not per.get(o_int, {}).get("ok", False):
            continue  # skip QC-failed orbit
        orbit_state = str(ts[state_coord].sel({orbit_dim: o}).values)
        # Get wet and Rc for this orbit
        orbit_wet = ts[wet_var].sel({orbit_dim: o}).astype(bool)
        orbit_rc = ts[rc_var].sel({orbit_dim: o})
        
        # Identify actual observation times (where Rc is not NaN)
        obs_mask = ~np.isnan(orbit_rc.values)
        
        # Select only actual observations for this orbit
        if obs_mask.any():
            # Get times and values only where orbit observed
            orbit_times = orbit_rc['Time'].values[obs_mask]
            orbit_wet_vals = orbit_wet.values[obs_mask]
            
            # Create Series indexed by time for this orbit
            wet_series = pd.Series(orbit_wet_vals, index=pd.to_datetime(orbit_times))
            
            if orbit_state == "ascending":
                wet_pm_list.append(wet_series)
            elif orbit_state == "descending":
                wet_am_list.append(wet_series)
    
    # Merge PM series: True if ANY ascending orbit was wet at that time
    if wet_pm_list:
        wet_pm_df = pd.concat(wet_pm_list, axis=1).fillna(False)
        wet_pm_series = wet_pm_df.any(axis=1)
    else:
        wet_pm_series = pd.Series([], dtype=bool)
    
    # Merge AM series: True if descending orbit was wet at that time  
    if wet_am_list:
        wet_am_df = pd.concat(wet_am_list, axis=1).fillna(False)
        wet_am_series = wet_am_df.any(axis=1)
    else:
        wet_am_series = pd.Series([], dtype=bool)
    
    # Keep wet_pm / wet_am sparse — only actual observation times, NO gap filling.
    # build_phase_intervals_ripe_start_both_stop_either_dry internally reindexes
    # to a union time grid and ffills, so gaps must be absent (NaN), not False.
    # Filling with False here would block ffill and prevent ripening from ever forming.
    if len(wet_pm_series) > 0:
        wet_pm = xr.DataArray(
            wet_pm_series.values.astype(bool),
            dims=['time'],
            coords={'time': wet_pm_series.index}
        )
    else:
        wet_pm = xr.DataArray(np.array([], dtype=bool), dims=['time'], coords={'time': []})

    if len(wet_am_series) > 0:
        wet_am = xr.DataArray(
            wet_am_series.values.astype(bool),
            dims=['time'],
            coords={'time': wet_am_series.index}
        )
    else:
        wet_am = xr.DataArray(np.array([], dtype=bool), dims=['time'], coords={'time': []})

    phases = build_phase_intervals_ripe_start_both_stop_either_dry(
        wet_pm=wet_pm,
        wet_am=wet_am,
        output_start=res["output_start"],   # pd.Timestamp or pd.NaT
        allow_ripe_end_two_dry=True,
    )

    return {
        "output_start": res["output_start"],
        "qc_reason": res["qc_reason"],
        "qc_ok": res["qc_ok"],
        "phases": phases,
        "diagnostics": res,
        "orbit_group": orbit_group,
        "ref_by_orbit": ref_by_orbit,
    }

def _da_bool_to_series(da_bool, time_name="Time"):
    s = da_bool.to_series()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    # ensure boolean
    s = s.fillna(False).astype(bool)
    return s

def output_start_lund_daily(
    aft_wet: xr.DataArray,
    morn_wet: xr.DataArray,
    ffill_days: int = 20,          # ~ one revisit + slack
    sustain_days: int = 14,        # require persistence
):
    """
    Returns a pandas Timestamp (or pd.NaT) for output start.

    - Resamples AM/PM wet series to daily.
    - Forward-fills wet state up to `ffill_days`.
    - Output starts at first day where both are True for `sustain_days` consecutive days.
    """
    sa = _da_bool_to_series(aft_wet)
    sm = _da_bool_to_series(morn_wet)

    # daily grid covering both
    start = min(sa.index.min(), sm.index.min()).normalize()
    end   = max(sa.index.max(), sm.index.max()).normalize()
    daily = pd.date_range(start, end, freq="D")

    # reindex to daily and forward-fill with limit
    da = sa.reindex(daily).ffill(limit=ffill_days).fillna(False)
    dm = sm.reindex(daily).ffill(limit=ffill_days).fillna(False)

    both = da & dm

    # find first run of sustain_days Trues
    run = both.rolling(sustain_days, min_periods=sustain_days).sum()
    hit = run[run >= sustain_days]
    if hit.empty:
        return pd.NaT

    # rolling marks the END of the first full window; convert to start day
    end_day = hit.index[0]
    start_day = end_day - pd.Timedelta(days=sustain_days - 1)
    return start_day



def output_start_lund_daily_window(aft_wet, morn_wet, window_days=10, sustain_days=14):
    """
    Expand sparse wet detections to a daily series using a symmetric ±window_days buffer.
    Output start = first day where BOTH expanded series are True for sustain_days consecutive days.
    """
    sa = _da_bool_to_series(aft_wet)   # True on acquisition days only
    sm = _da_bool_to_series(morn_wet)

    start = min(sa.index.min(), sm.index.min()).normalize()
    end   = max(sa.index.max(), sm.index.max()).normalize()
    daily = pd.date_range(start, end, freq="D")

    def expand_true_days(s, window):
        # s is boolean series on acquisition dates; expand each True to ±window days
        expanded = pd.Series(False, index=daily)
        true_days = s[s].index.normalize()
        for d in true_days:
            lo = d - pd.Timedelta(days=window)
            hi = d + pd.Timedelta(days=window)
            expanded.loc[(expanded.index >= lo) & (expanded.index <= hi)] = True
        return expanded

    da = expand_true_days(sa, window_days)
    dm = expand_true_days(sm, window_days)

    both = da & dm

    run = both.rolling(sustain_days, min_periods=sustain_days).sum()
    hit = run[run >= sustain_days]
    if hit.empty:
        return pd.NaT

    end_day = hit.index[0]
    return end_day - pd.Timedelta(days=sustain_days - 1)


def compute_W(lia, k=0.5, theta1=20, theta2=45):
    W = xr.where(lia < theta1, 1,
                 k * (1 + ((theta2 - lia) / (theta2 - theta1))))
    W = xr.where(lia > theta2, k, W)
    return W.clip(0, 1)

def build_point_timeseries_all_orbits_from_files(
    path, cues_y, cues_x, orbits=(64, 137, 144),
    k=0.5, theta1=20, theta2=45, wet_thresh_db=-2.0,
):
    """
    For your per-date zarr layout:
      - each file contains ratio_images(sat:relative_orbit, band, y, x)
      - time is a coord on sat:relative_orbit: ds['time'](sat:relative_orbit)

    Returns dict[orbit] -> xr.Dataset with dims (Time,)
      dvv, dvh, Rc, wet
    """
    # store as python lists then convert
    rows = {int(o): {"Time": [], "dvv": [], "dvh": [], "Rc": [], "wet": []} for o in orbits}

    for file in sorted(os.listdir(path)):
        if file.startswith("."):
            continue

        ds = xr.open_zarr(os.path.join(path, file)).rio.write_crs("EPSG:32611")

        # compute W and Rc on grid (still per-orbit)
        W = compute_W(ds.local_incidence_angle, k=k, theta1=theta1, theta2=theta2)
        dvv = ds.ratio_images.sel(band="vv")
        dvh = ds.ratio_images.sel(band="vh")
        Rc  = W * dvh + (1 - W) * dvv

        # nearest pixel indices once per file
        idy, idx = sm_intersect_rectilinear(ds.y.values, ds.x.values, cues_y, cues_x)

        for o in orbits:
            o = int(o)
            if o not in ds["sat:relative_orbit"].values:
                continue

            # timestamp for this orbit in this file
            t = ds["time"].sel(**{"sat:relative_orbit": o}).values
            t = np.datetime64(t, "ns")

            # scalar values at point for this orbit
            dvv_val = float(dvv.sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            dvh_val = float(dvh.sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            Rc_val  = float(Rc .sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            wet_val = bool(Rc_val < wet_thresh_db)

            rows[o]["Time"].append(t)
            rows[o]["dvv"].append(dvv_val)
            rows[o]["dvh"].append(dvh_val)
            rows[o]["Rc"].append(Rc_val)
            rows[o]["wet"].append(wet_val)

    # build xarray datasets per orbit
    out = {}
    for o, d in rows.items():
        if len(d["Time"]) == 0:
            out[o] = None
            continue

        # sort by time and drop duplicates
        df = pd.DataFrame(d).sort_values("Time").drop_duplicates("Time")
        out[o] = xr.Dataset(
            {
                "dvv": xr.DataArray(df["dvv"].values, dims=("Time",), coords={"Time": df["Time"].values}),
                "dvh": xr.DataArray(df["dvh"].values, dims=("Time",), coords={"Time": df["Time"].values}),
                "Rc":  xr.DataArray(df["Rc"].values,  dims=("Time",), coords={"Time": df["Time"].values}),
                "wet": xr.DataArray(df["wet"].values, dims=("Time",), coords={"Time": df["Time"].values}),
            }
        )

    return out

def ts_dict_to_ds(ts_dict, orbit_dim="sat:relative_orbit", state_coord="sat:orbit_state"):
    # Ensure deterministic orbit order
    orbits = sorted(list(ts_dict.keys()))

    # concat into (sat:relative_orbit, Time)
    ds = xr.concat(
        [ts_dict[o].expand_dims({orbit_dim: [int(o)]}) for o in orbits],
        dim=orbit_dim,
        join="outer",
    )

    # Assign orbit_state (CUES: 64/137 = ascending, 144 = descending)
    orbit_state_map = {64: "ascending", 137: "ascending", 144: "descending"}
    ds = ds.assign_coords(
        {
            state_coord: (
                orbit_dim,
                [orbit_state_map.get(int(o), None) for o in ds[orbit_dim].values],
            )
        }
    )

    return ds



@dataclass
class WeightedMinimaCfg:
    # --- used to BUILD the Rc point timeseries from files ---
    k: float = 0.5
    theta1: float = 20
    theta2: float = 45
    wet_thresh_db: float = -2.0

    # --- used by Marin minima QC (same dataclass you already have) ---
    # (we’ll pass this into output_start_marin_minima_qc via cfg_marin)
    cfg_marin: "MarinS1VVCfg" = None


def run_s1_rc_minima_year(
    path_year: str,
    cues_y: float,
    cues_x: float,
    year: int,
    cfg_w: WeightedMinimaCfg,
    *,
    orbits=(64, 137, 144),
    showOutput: bool = False,
):
    """
    Marin-style minima+QC, but on Rc (weighted VV/VH).
    Output onset comes from Rc minima across AM/PM orbit groups, NOT sustained BOTH-wet logic.
    Returns a dict like run_s1_vv_year.
    """

    # 1) build point timeseries per orbit from files
    ts_dict = build_point_timeseries_all_orbits_from_files(
        path_year,
        cues_y,
        cues_x,
        orbits=list(orbits),
        k=cfg_w.k,
        theta1=cfg_w.theta1,
        theta2=cfg_w.theta2,
        wet_thresh_db=cfg_w.wet_thresh_db,
        # showOutput=showOutput,
    )

    # 2) dict -> xr.Dataset with dims (sat:relative_orbit, Time)
    ts_ds = ts_dict_to_ds(ts_dict)

    # 3) run the Rc-minima QC pipeline
    if cfg_w.cfg_marin is None:
        cfg_marin = MarinS1VVCfg(min_stability_days=28, max_group_spread_days=28)
    else:
        cfg_marin = cfg_w.cfg_marin

    out = run_s1_rc_year_from_ts(ts_ds, year, cfg_marin)

    # out already has: output_start, qc_ok, qc_reason, phases, diagnostics, ref_by_orbit
    return out

def build_point_timeseries_all_orbits_from_files2(
    path, cues_y, cues_x, orbits=(64, 137, 144),
    k=0.5, theta1=20, theta2=45, wet_thresh_db=-2.0,
):
    """
    For your per-date zarr layout:
      - each file contains ratio_images(sat:relative_orbit, band, y, x)
      - time is a coord on sat:relative_orbit: ds['time'](sat:relative_orbit)

    Returns dict[orbit] -> xr.Dataset with dims (Time,)
      dvv, dvh, Rc, wet
    """
    # store as python lists then convert
    rows = {int(o): {"Time": [], "dvv": [], "dvh": [], "Rc": [], "wet": []} for o in orbits}

    for file in sorted(os.listdir(path)):
        if file.startswith("."):
            continue

        ds = xr.open_zarr(os.path.join(path, file)).rio.write_crs("EPSG:32611")

        # compute W and Rc on grid (still per-orbit)
        W = compute_W(ds.local_incidence_angle, k=k, theta1=theta1, theta2=theta2)
        dvv = ds.ratio_images.sel(band="vv")
        dvh = ds.ratio_images.sel(band="vh")
        Rc  = W * dvh + (1 - W) * dvv

        # nearest pixel indices once per file
        idy, idx = sm_intersect_rectilinear(ds.y.values, ds.x.values, cues_y, cues_x)

        for o in orbits:
            o = int(o)
            if o not in ds["sat:relative_orbit"].values:
                continue

            # timestamp for this orbit in this file
            t = ds["time"].sel(**{"sat:relative_orbit": o}).values
            t = np.datetime64(t, "ns")

            # scalar values at point for this orbit
            dvv_val = float(dvv.sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            dvh_val = float(dvh.sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            Rc_val  = float(Rc .sel(**{"sat:relative_orbit": o}).isel(y=idy, x=idx).values)
            wet_val = bool(Rc_val < wet_thresh_db)

            rows[o]["Time"].append(t)
            rows[o]["dvv"].append(dvv_val)
            rows[o]["dvh"].append(dvh_val)
            rows[o]["Rc"].append(Rc_val)
            rows[o]["wet"].append(wet_val)

    # build xarray datasets per orbit
    out = {}
    for o, d in rows.items():
        if len(d["Time"]) == 0:
            out[o] = None
            continue

        # sort by time and drop duplicates
        df = pd.DataFrame(d).sort_values("Time").drop_duplicates("Time")
        out[o] = xr.Dataset(
            {
                "dvv": xr.DataArray(df["dvv"].values, dims=("Time",), coords={"Time": df["Time"].values}),
                "dvh": xr.DataArray(df["dvh"].values, dims=("Time",), coords={"Time": df["Time"].values}),
                "Rc":  xr.DataArray(df["Rc"].values,  dims=("Time",), coords={"Time": df["Time"].values}),
                "wet": xr.DataArray(df["wet"].values, dims=("Time",), coords={"Time": df["Time"].values}),
            }
        )

    return out

def run_vv_spatial(bscatter_ds: xr.Dataset,
                   reference_ds: xr.Dataset,
                   year: int,
                   cfg,
                   start_date: str = None,
                   end_date: str = None) -> xr.Dataset:
    """
    Runs the S1-VV (Marin minima) method over the full spatial domain.

    Loops over each (y, x) pixel and calls melt_phases.run_s1_vv_year, which
    is identical to what fig_2 does at the CUES point.

    Parameters
    ----------
    bscatter_ds  : Dataset from load_data.load_sentinel_backscatter (clipped to mammoth)
    reference_ds : Dataset from load_data.load_sentinel_reference   (clipped to mammoth)
    year         : int — water year
    cfg          : melt_phases.MarinS1VVCfg
    start_date   : optional ISO string for daily output grid start (default: {year}-01-01)
    end_date     : optional ISO string for daily output grid end   (default: {year}-07-01)

    Returns
    -------
    xr.Dataset with:
        phase_name   (time, y, x)  — daily phase string (dry/moistening/ripening/output)
        output_start (y, x)        — Marin output onset date (NaT if QC failed)
        qc_ok        (y, x)        — bool, True if QC passed
    """
    if start_date is None:
        start_date = f"{year}-01-01"
    if end_date is None:
        end_date = f"{year}-07-01"

    daily_index = pd.date_range(start_date, end_date, freq="D")

    ys = bscatter_ds.y.values
    xs = bscatter_ds.x.values
    ny, nx = len(ys), len(xs)

    # Pre-allocate output arrays
    phase_arr   = np.full((len(daily_index), ny, nx), "na", dtype="U12")
    out_start   = np.full((ny, nx), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    qc_ok_arr   = np.zeros((ny, nx), dtype=bool)

    PHASE_CODES = {"dry": 0, "moistening": 1, "ripening": 2, "output": 3, "na": -1}

    for iy in tqdm(range(ny), desc=f"VV spatial {year}"):
        for ix in range(nx):
            try:
                result = run_s1_vv_year(
                    bscatter_ds, reference_ds, iy, ix, year, cfg
                )
            except Exception:
                continue

            qc_ok_arr[iy, ix] = result["qc_ok"]

            if result["qc_ok"] and result.get("output_start") is not None:
                t_out = result["output_start"]
                if not pd.isnull(t_out):
                    out_start[iy, ix] = np.datetime64(t_out, "ns")

            # convert interval-based phases to per-day series
            if result["phases"] is not None:
                try:
                    day_s = to_daily_phase_series_local(result["phases"], start_date, end_date)
                    for di, d in enumerate(daily_index):
                        if d in day_s.index:
                            phase_arr[di, iy, ix] = day_s.loc[d]
                except Exception:
                    pass

    return xr.Dataset(
        {
            "phase_name": xr.DataArray(
                phase_arr,
                dims=("time", "y", "x"),
                coords={"time": daily_index, "y": ys, "x": xs},
            ),
            "output_start": xr.DataArray(
                out_start,
                dims=("y", "x"),
                coords={"y": ys, "x": xs},
            ),
            "qc_ok": xr.DataArray(
                qc_ok_arr,
                dims=("y", "x"),
                coords={"y": ys, "x": xs},
            ),
        }
    )


def to_daily_phase_series_local(phases_ds: xr.Dataset,
                                start: str,
                                end: str) -> pd.Series:
    """
    Convert an interval-style phases Dataset (interval_start, interval_end, phase_name)
    to a daily pd.Series — mirrors the logic inside fig_2's to_daily_phase_series().

    Parameters
    ----------
    phases_ds : xr.Dataset with coords 'interval_start', variable 'phase_name'
                (and optionally 'interval_end')
    start, end : ISO date strings

    Returns
    -------
    pd.Series indexed by date, values are phase name strings.
    """
    idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    out = pd.Series("na", index=idx, name="phase", dtype="object")

    if phases_ds is None:
        return out

    # Handle phase DataArrays with time dimension (returned by build_phase_intervals_*)
    if "Time" in phases_ds.dims:
        t = pd.to_datetime(phases_ds["Time"].values)
        names = phases_ds["phase_name"].values if "phase_name" in phases_ds else None
        if names is not None:
            for ti, ph in zip(t, names):
                d = pd.Timestamp(ti).normalize()
                if d in idx:
                    out.loc[d] = str(ph)
        return out

    # Handle interval-style (interval_start coord)
    if "interval_start" in phases_ds.coords:
        starts = pd.to_datetime(phases_ds["interval_start"].values)
        ends   = pd.to_datetime(phases_ds["interval_end"].values) if "interval_end" in phases_ds else starts
        ph_vals = phases_ds["phase_name"].values if "phase_name" in phases_ds else phases_ds["phase_id"].values
        order = np.argsort(starts)
        starts, ends, ph_vals = starts[order], ends[order], ph_vals[order]
        for st, en, ph in zip(starts, ends, ph_vals):
            if pd.isna(st):
                continue
            d0 = max(st.normalize(), idx[0])
            d1 = min(en.normalize() if not pd.isna(en) else idx[-1], idx[-1])
            if d1 >= d0:
                out.loc[d0:d1] = str(ph)
    return out

def build_spatial_rc_from_files(path: str,
                                clip_gdf,
                                orbits=(64, 137, 144),
                                k: float = 0.5,
                                theta1: float = 20,
                                theta2: float = 45,
                                wet_thresh_db: float = -2.0) -> dict:
    """
    Spatial version of melt_phases.build_point_timeseries_all_orbits_from_files.

    Opens each zarr file in `path`, clips to `clip_gdf`, computes the weighted
    Rc = W*dvh + (1-W)*dvv across the full (y, x) spatial domain.

    Parameters
    ----------
    path         : str — directory containing per-acquisition zarr stores
                   (e.g. `.../melt_thresh/data/datasets/`)
    clip_gdf     : GeoDataFrame used to clip each acquisition
    orbits       : tuple of orbit numbers to include
    k, theta1, theta2, wet_thresh_db : Weighted method parameters

    Returns
    -------
    dict[orbit_int] -> xr.Dataset with dims (Time, y, x):
        dvv  : VV ratio image
        dvh  : VH ratio image
        Rc   : weighted metric
        wet  : bool (Rc < wet_thresh_db)
    """
    orbit_rows = {int(o): {"Time": [], "dvv": [], "dvh": [], "Rc": [], "wet": []} for o in orbits}
    orbit_yx   = {}  # will hold (y_values, x_values) from first clipped file per orbit

    files = sorted(f for f in os.listdir(path) if not f.startswith("."))

    for file in tqdm(files, desc="Loading zarr files"):
        fpath = os.path.join(path, file)
        try:
            ds = xr.open_zarr(fpath).rio.write_crs("EPSG:32611")
            ds_clip = ds.rio.clip(clip_gdf.geometry, all_touched=True)
        except Exception as e:
            print(f"  Skipping {file}: {e}")
            continue

        # Compute W and Rc spatially across full domain
        lia = ds_clip.local_incidence_angle  # (sat:relative_orbit, y, x) or (y, x)
        W = compute_W(lia, k=k, theta1=theta1, theta2=theta2)

        dvv = ds_clip.ratio_images.sel(band="vv")
        dvh = ds_clip.ratio_images.sel(band="vh")
        Rc  = W * dvh + (1 - W) * dvv   # (sat:relative_orbit, y, x) or broadcast

        for o in orbits:
            o = int(o)
            if o not in ds_clip["sat:relative_orbit"].values:
                continue

            sel = {"sat:relative_orbit": o}
            t   = ds_clip["time"].sel(**sel).values          # scalar timestamp
            t64 = np.datetime64(t, "ns")

            dvv_o = dvv.sel(**sel)   # (y, x)
            dvh_o = dvh.sel(**sel)   # (y, x)
            Rc_o  = Rc.sel(**sel) if "sat:relative_orbit" in Rc.dims else Rc   # (y, x)
            wet_o = (Rc_o < wet_thresh_db)                   # (y, x) bool

            # Store y/x coords on first encounter
            if o not in orbit_yx:
                orbit_yx[o] = (dvv_o.y.values, dvv_o.x.values)

            orbit_rows[o]["Time"].append(t64)
            orbit_rows[o]["dvv"].append(dvv_o.values)
            orbit_rows[o]["dvh"].append(dvh_o.values)
            orbit_rows[o]["Rc"].append(Rc_o.values)
            orbit_rows[o]["wet"].append(wet_o.values.astype(bool))

    # Build xr.Dataset per orbit: (Time, y, x)
    out = {}
    for o, rows in orbit_rows.items():
        if len(rows["Time"]) == 0:
            out[o] = None
            continue

        # sort by time, remove duplicates
        times = np.array(rows["Time"], dtype="datetime64[ns]")
        order = np.argsort(times)
        _, uniq = np.unique(times[order], return_index=True)
        idx = order[uniq]

        ys, xs = orbit_yx[o]

        def stack(key):
            arr = np.stack([rows[key][i] for i in idx], axis=0)  # (Time, y, x)
            return xr.DataArray(arr, dims=("Time", "y", "x"),
                                coords={"Time": times[idx], "y": ys, "x": xs})

        out[o] = xr.Dataset({
            "dvv": stack("dvv"),
            "dvh": stack("dvh"),
            "Rc":  stack("Rc"),
            "wet": stack("wet"),
        })

    return out

def run_weighted_spatial(rc_spatial: dict,
                         year: int,
                         cfg_w,
                         start_date: str = None,
                         end_date: str = None) -> xr.Dataset:
    """
    Runs the S1-Weighted (Rc) phase classification over the full spatial domain.

    Uses run_s1_rc_year_from_ts per pixel, which returns phases in the same
    interval format as run_s1_vv_year — ensuring identical logic to fig_2.

    Parameters
    ----------
    rc_spatial : dict[orbit] -> xr.Dataset(dvv, dvh, Rc, wet; dims Time,y,x)
                 as returned by build_spatial_rc_from_files
    year       : int — water year
    cfg_w      : melt_phases.WeightedMinimaCfg
    start_date, end_date : ISO strings for daily output grid

    Returns
    -------
    xr.Dataset with:
        phase_name   (time, y, x)  — daily phase string (interval forward-filled)
        output_start (y, x)        — Marin output onset (NaT if QC failed)
        qc_ok        (y, x)        — bool
    """
    if start_date is None:
        start_date = f"{year}-01-01"
    if end_date is None:
        end_date = f"{year}-07-01"

    daily_index = pd.date_range(start_date, end_date, freq="D")

    # Determine spatial grid from first valid orbit dataset
    ref_ds = next((v for v in rc_spatial.values() if v is not None), None)
    if ref_ds is None:
        raise ValueError("All orbits returned None — check zarr path.")

    ys = ref_ds.y.values
    xs = ref_ds.x.values
    ny, nx = len(ys), len(xs)

    # Pre-allocate
    phase_arr  = np.full((len(daily_index), ny, nx), "na", dtype="U12")
    out_start  = np.full((ny, nx), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    qc_ok_arr  = np.zeros((ny, nx), dtype=bool)

    cfg_marin = cfg_w.cfg_marin if cfg_w.cfg_marin is not None else melt_phases.MarinS1VVCfg()

    for iy in tqdm(range(ny), desc=f"Weighted spatial {year}"):
        for ix in range(nx):
            # Build point timeseries dict[orbit] -> xr.Dataset(Time,) from spatial slice
            ts_pt = {}
            for o, ds in rc_spatial.items():
                if ds is None:
                    ts_pt[o] = None
                    continue
                pt = ds.isel(y=iy, x=ix)   # (Time,) Dataset
                ts_pt[o] = xr.Dataset({
                    "dvv": pt["dvv"],
                    "dvh": pt["dvh"],
                    "Rc":  pt["Rc"],
                    "wet": pt["wet"],
                })

            try:
                # Stack orbits into (sat:relative_orbit, Time) structure — assigns
                # sat:orbit_state (ascending/descending) per orbit, matching fig_2
                ts_ds = ts_dict_to_ds(ts_pt)

                # Run the full Rc-Marin pipeline — returns phases in interval format,
                # identical to what run_s1_vv_year returns (same logic as fig_2)
                marin_out = run_s1_rc_year_from_ts(ts_ds, year, cfg_marin)

                qc_ok_arr[iy, ix] = marin_out["qc_ok"]

                t_out = marin_out.get("output_start", pd.NaT)
                if not pd.isnull(t_out):
                    out_start[iy, ix] = np.datetime64(t_out, "ns")

                # Convert interval-based phases to daily — same path as run_vv_spatial
                if marin_out["phases"] is not None:
                    day_s = to_daily_phase_series_local(
                        marin_out["phases"], start_date, end_date
                    )
                    for di, d in enumerate(daily_index):
                        if d in day_s.index:
                            phase_arr[di, iy, ix] = day_s.loc[d]

            except Exception:
                pass   # pixel stays "na" — masked / edge pixels

    return xr.Dataset(
        {
            "phase_name": xr.DataArray(
                phase_arr,
                dims=("time", "y", "x"),
                coords={"time": daily_index, "y": ys, "x": xs},
            ),
            "output_start": xr.DataArray(
                out_start,
                dims=("y", "x"),
                coords={"y": ys, "x": xs},
            ),
            "qc_ok": xr.DataArray(
                qc_ok_arr,
                dims=("y", "x"),
                coords={"y": ys, "x": xs},
            ),
        }
    )





    
