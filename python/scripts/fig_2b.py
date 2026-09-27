import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def onset_date_consecutive(phase_s: pd.Series,
                           phase: str,
                           start=None,
                           end=None,
                           min_consecutive_days: int = 3):
    """
    First day of the first run of `phase` lasting >= min_consecutive_days.
    Returns pd.Timestamp or None.
    """
    s = phase_s.copy()
    s.index = pd.to_datetime(s.index)

    if start is not None:
        s = s.loc[pd.to_datetime(start):]
    if end is not None:
        s = s.loc[:pd.to_datetime(end)]

    is_phase = (s == phase)
    hits = is_phase.rolling(min_consecutive_days, min_periods=min_consecutive_days).sum()
    cand = hits[hits == min_consecutive_days]
    if cand.empty:
        return None
    end_day = cand.index[0]
    return end_day - pd.Timedelta(days=min_consecutive_days - 1)


def percent_agreement(a: pd.Series, b: pd.Series, include_na: bool = False) -> float:
    """
    Percent of days where a == b after aligning to common index.
    If include_na=False, drops days where either is 'na' (or NaN).
    Returns percent [0..100].
    """
    aa, bb = a.align(b, join="inner")
    aa = aa.astype("object")
    bb = bb.astype("object")

    mask = (~aa.isna()) & (~bb.isna())
    if not include_na:
        mask &= (aa != "na") & (bb != "na")

    if mask.sum() == 0:
        return np.nan

    return 100.0 * (aa[mask] == bb[mask]).mean()

def build_onset_table(phases_by_year: dict,
                      phases=("ripening", "output"),
                      start=None,
                      end=None,
                      min_consecutive_days=3):
    """
    Returns a tidy DataFrame with onset dates per year/method/phase.
    """
    rows = []
    for yr, methods in phases_by_year.items():
        for method_name, s in methods.items():
            for ph in phases:
                dt = onset_date_consecutive(s, ph, start=start, end=end,
                                            min_consecutive_days=min_consecutive_days)
                rows.append({
                    "year": int(yr),
                    "method": str(method_name),
                    "phase": str(ph),
                    "onset": pd.to_datetime(dt) if dt is not None else pd.NaT
                })
    df = pd.DataFrame(rows)
    return df

def build_agreement_matrices(phases_by_year: dict,
                             method_order=None,
                             include_na=False,
                             pooled=False):
    """
    Returns:
      - if pooled=False: dict[year] -> (methods, matrix)
      - if pooled=True: (methods, matrix) pooled across years (concatenate days)
    """
    # determine method order
    if method_order is None:
        # union of all methods across years
        all_methods = sorted({m for yr in phases_by_year for m in phases_by_year[yr].keys()})
        method_order = all_methods

    if pooled:
        pooled_series = {m: [] for m in method_order}
        for yr, methods in phases_by_year.items():
            for m in method_order:
                if m in methods:
                    pooled_series[m].append(methods[m])
        # concatenate
        pooled_series = {
            m: pd.concat(pooled_series[m]).sort_index() if len(pooled_series[m]) else pd.Series(dtype="object")
            for m in method_order
        }

        mat = np.full((len(method_order), len(method_order)), np.nan)
        for i, mi in enumerate(method_order):
            for j, mj in enumerate(method_order):
                mat[i, j] = percent_agreement(pooled_series[mi], pooled_series[mj], include_na=include_na)
        return method_order, mat

    mats = {}
    for yr, methods in phases_by_year.items():
        mat = np.full((len(method_order), len(method_order)), np.nan)
        for i, mi in enumerate(method_order):
            for j, mj in enumerate(method_order):
                if mi in methods and mj in methods:
                    mat[i, j] = percent_agreement(methods[mi], methods[mj], include_na=include_na)
        mats[int(yr)] = (method_order, mat)

    return mats

def plot_summary_A_B(phases_by_year: dict,
                     method_order=None,
                     window_start=None,
                     window_end=None,
                     onset_phases=("ripening", "output"),
                     min_consecutive_days=3,
                     include_na_agreement=False,
                     pooled_agreement=True,
                     figsize=(12, 5),
                     dpi=200):
    """
    Creates a 2-panel figure:
      A) onset timing spread + method dots (for each year)
      B) agreement matrix heatmap (% agreement)
    """
    # Method order
    if method_order is None:
        method_order = sorted({m for yr in phases_by_year for m in phases_by_year[yr].keys()})

    # Build onset table
    onset_df = build_onset_table(
        phases_by_year,
        phases=onset_phases,
        start=window_start,
        end=window_end,
        min_consecutive_days=min_consecutive_days,
    )

    # Convert onset dates to day-of-water-year (optional) or DOY; DOY is simplest
    onset_df["doy"] = onset_df["onset"].dt.dayofyear

    # Agreement matrix
    if pooled_agreement:
        methods, agr = build_agreement_matrices(
            phases_by_year, method_order=method_order,
            include_na=include_na_agreement, pooled=True
        )
    else:
        # average across years if not pooled
        mats = build_agreement_matrices(
            phases_by_year, method_order=method_order,
            include_na=include_na_agreement, pooled=False
        )
        methods = method_order
        stack = []
        for yr, (_, mat) in mats.items():
            stack.append(mat)
        agr = np.nanmean(np.stack(stack, axis=0), axis=0)

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=len(onset_phases), ncols=2, width_ratios=[1.7, 1.0], wspace=0.35, hspace=0.35)

    # Panel A axes (one per phase)
    axA = []
    for r in range(len(onset_phases)):
        axA.append(fig.add_subplot(gs[r, 0]))

    # Panel B axis (heatmap spans rows)
    axB = fig.add_subplot(gs[:, 1])

    # ---------------- Panel A: onset spread + dots ----------------
    years = sorted(phases_by_year.keys())
    y_positions = np.arange(len(years))

    # method markers (matplotlib defaults); no custom colors required
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(method_order)}

    for r, ph in enumerate(onset_phases):
        ax = axA[r]
        dfp = onset_df[onset_df["phase"] == ph].copy()

        # For each year, plot min-max bar (spread) and method dots
        for yi, yr in enumerate(years):
            sub = dfp[dfp["year"] == int(yr)].dropna(subset=["onset"])
            if sub.empty:
                continue

            # spread
            xmin = sub["onset"].min()
            xmax = sub["onset"].max()
            ax.hlines(yi, xmin, xmax, linewidth=6, alpha=0.15)

            # method dots
            for m in method_order:
                row = sub[sub["method"] == m]
                if row.empty:
                    continue
                ax.scatter(row["onset"].iloc[0], yi, s=35, marker=marker_map[m])

        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(y) for y in years])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title(f"A) {ph.capitalize()} onset (spread across methods)", fontweight="bold" if r == 0 else None)

        # x-limits based on window
        if window_start and window_end:
            ax.set_xlim(pd.to_datetime(window_start), pd.to_datetime(window_end))

        if r == len(onset_phases) - 1:
            ax.set_xlabel("Date")

    # Add one shared legend for methods (Panel A)
    legend_handles = []
    for m in method_order:
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker_map[m], linestyle="None", label=m)
        )
    axA[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.35),
                 ncol=min(4, len(method_order)), frameon=True, fontsize=8, title="Method")

    # ---------------- Panel B: agreement heatmap ----------------
    im = axB.imshow(agr, vmin=0, vmax=100, aspect="auto")
    axB.set_title("B) Pairwise agreement (% days)", fontweight="bold")
    axB.set_xticks(np.arange(len(methods)))
    axB.set_yticks(np.arange(len(methods)))
    axB.set_xticklabels(methods, rotation=45, ha="right")
    axB.set_yticklabels(methods)

    # annotate cells
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isfinite(agr[i, j]):
                axB.text(j, i, f"{agr[i, j]:.0f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement (%)")

    # note about NA handling
    note = "Agreement excludes NA days" if not include_na_agreement else "Agreement includes NA days"
    axB.text(0.0, -0.12, note, transform=axB.transAxes, fontsize=8)

    plt.tight_layout()
    return fig, (axA, axB), onset_df, (methods, agr)

def plot_summary_A_B_doy(
    phases_by_year: dict,
    method_order,
    onset_phases=("moistening", "ripening", "output"),
    min_consecutive_days=3,
    window_mmdd=("01-01", "06-01"),
    include_na_agreement=False,
    pooled_agreement=True,
    figsize=(12, 6),
    dpi=200,
    method_colors=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- helpers ---
    def onset_date_consecutive(phase_s, phase, start, end, k):
        s = phase_s.copy()
        s.index = pd.to_datetime(s.index)
        s = s.loc[start:end]
        is_phase = (s == phase)
        hits = is_phase.rolling(k, min_periods=k).sum()
        cand = hits[hits == k]
        if cand.empty:
            return None
        return cand.index[0] - pd.Timedelta(days=k - 1)

    def percent_agreement(a, b, include_na):
        aa, bb = a.align(b, join="inner")
        aa = aa.astype("object")
        bb = bb.astype("object")
        mask = (~aa.isna()) & (~bb.isna())
        if not include_na:
            mask &= (aa != "na") & (bb != "na")
        if mask.sum() == 0:
            return np.nan, 0
        return 100.0 * (aa[mask] == bb[mask]).mean(), int(mask.sum())

    years = sorted(phases_by_year.keys())
    y_positions = np.arange(len(years))

    # Default colors if none provided
    if method_colors is None:
        # Matplotlib default cycle colors; stable by method order
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        method_colors = {m: default_cycle[i % len(default_cycle)] for i, m in enumerate(method_order)}

    # Marker mapping
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(method_order)}

    # ---- Build onset table (in DOY) ----
    onset_rows = []
    for yr in years:
        start = pd.to_datetime(f"{yr}-{window_mmdd[0]}")
        end   = pd.to_datetime(f"{yr}-{window_mmdd[1]}")
        for m in method_order:
            if m not in phases_by_year[yr]:
                continue
            s = phases_by_year[yr][m]
            for ph in onset_phases:
                dt = onset_date_consecutive(s, ph, start, end, min_consecutive_days)
                onset_rows.append({
                    "year": yr,
                    "method": m,
                    "phase": ph,
                    "onset": pd.to_datetime(dt) if dt is not None else pd.NaT,
                    "doy": (pd.to_datetime(dt).dayofyear if dt is not None else np.nan),
                })
    onset_df = pd.DataFrame(onset_rows)
    if onset_df.empty:
        onset_df = pd.DataFrame(columns=["year", "method", "phase", "onset", "doy"])

    # ---- Agreement matrix pooled across years in the same seasonal window ----
    pooled = {m: [] for m in method_order}
    for yr in years:
        start = pd.to_datetime(f"{yr}-{window_mmdd[0]}")
        end   = pd.to_datetime(f"{yr}-{window_mmdd[1]}")
        for m in method_order:
            if m in phases_by_year[yr]:
                pooled[m].append(phases_by_year[yr][m].loc[start:end])

    pooled = {
        m: pd.concat(pooled[m]).sort_index() if len(pooled[m]) else pd.Series(dtype="object")
        for m in method_order
    }

    agr = np.full((len(method_order), len(method_order)), np.nan)
    n_ov = np.zeros((len(method_order), len(method_order)), dtype=int)

    for i, mi in enumerate(method_order):
        for j, mj in enumerate(method_order):
            a, n = percent_agreement(pooled[mi], pooled[mj], include_na_agreement)
            agr[i, j] = a
            n_ov[i, j] = n

    # ---- Plot layout ----
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=len(onset_phases), ncols=2, width_ratios=[1.7, 1.0],
                          wspace=0.35, hspace=0.45)

    axA = [fig.add_subplot(gs[r, 0]) for r in range(len(onset_phases))]
    axB = fig.add_subplot(gs[:, 1])

    # ---- Panel A ----
    for r, ph in enumerate(onset_phases):
        ax = axA[r]
        dfp = onset_df[onset_df["phase"] == ph].dropna(subset=["doy"])

        for yi, yr in enumerate(years):
            sub = dfp[dfp["year"] == yr]
            if sub.empty:
                continue

            xmin, xmax = sub["doy"].min(), sub["doy"].max()
            ax.hlines(yi, xmin, xmax, linewidth=3, alpha=0.08, color="k")

            for m in method_order:
                row = sub[sub["method"] == m]
                if row.empty:
                    continue
                ax.scatter(
                    row["doy"].iloc[0], yi,
                    s=55,
                    marker=marker_map[m],
                    color=method_colors[m],
                    edgecolor="k",
                    linewidth=0.4,
                    zorder=3
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(y) for y in years])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)

        ax.set_title(f"A) {ph.capitalize()} onset (DOY; spread across methods)",
                     fontweight="bold" if r == 0 else None)

        if r == len(onset_phases) - 1:
            ax.set_xlabel("Day of year (DOY)")

    # ---- Figure-level legend (clean) ----
    legend_handles = [
        plt.Line2D([0], [0],
                   marker=marker_map[m], linestyle="None",
                   markerfacecolor=method_colors[m],
                   markeredgecolor="k",
                   markersize=7,
                   label=m)
        for m in method_order
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.42, 1.02),
               ncol=len(method_order), frameon=True, title="Method")

    # ---- Panel B ----
    im = axB.imshow(agr, vmin=40, vmax=100, aspect="auto")
    axB.set_title("B) Pairwise agreement (% days; pooled Jan–Jun)", fontweight="bold")
    axB.set_xticks(np.arange(len(method_order)))
    axB.set_yticks(np.arange(len(method_order)))
    axB.set_xticklabels(method_order, rotation=30, ha="right")
    axB.set_yticklabels(method_order)

    for i in range(len(method_order)):
        for j in range(len(method_order)):
            if np.isfinite(agr[i, j]):
                axB.text(j, i, f"{agr[i, j]:.0f}\n(n={n_ov[i,j]})",
                         ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement (%)")

    note = "Agreement excludes NA days" if not include_na_agreement else "Agreement includes NA days"
    axB.text(0.0, -0.10, note, transform=axB.transAxes, fontsize=8)

    plt.tight_layout()
    return fig, (axA, axB), onset_df, (method_order, agr, n_ov)


# DEFAULT_PHASE_MAP = {
#     "moist": "moistening",
#     "moisten": "moistening",
#     "ripe": "ripening",
#     "ripen": "ripening",
#     "wet": "output",
#     "melt": "output",
#     "nan": "na",
#     "none": "na",
# }

# ALLOWED = {"dry", "moistening", "ripening", "output", "na"}

# def normalize_phase_series(s: pd.Series, mapping=None, fill_value="na") -> pd.Series:
#     if mapping is None:
#         mapping = DEFAULT_PHASE_MAP
#     s2 = s.astype("object").apply(lambda v: str(v).lower().strip() if pd.notna(v) else np.nan)
#     s2 = s2.replace(mapping).fillna(fill_value)
#     return s2

# def sm_3h_to_daily_majority(ds, phase_var="phase_name_3h") -> pd.Series:
#     s = ds[phase_var].to_series()
#     s.index = pd.to_datetime(s.index)
#     daily = s.groupby(s.index.normalize()).agg(
#         lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else np.nan
#     )
#     daily.name = "phase"
#     return daily

# def s1_intervals_to_daily_fill(ds, start, end,
#                               start_var="interval_start",
#                               end_var="interval_end",
#                               phase_var="phase_name",
#                               fill_value="na") -> pd.Series:
#     start = pd.to_datetime(start)
#     end = pd.to_datetime(end)
#     idx = pd.date_range(start, end, freq="D")

#     starts = pd.to_datetime(ds[start_var].values)
#     ends   = pd.to_datetime(ds[end_var].values)
#     phases = pd.Series(ds[phase_var].values).values

#     order = np.argsort(starts.values)
#     starts, ends, phases = starts[order], ends[order], phases[order]

#     out = pd.Series(index=idx, dtype="object")
#     for st, en, ph in zip(starts, ends, phases):
#         if pd.isna(st) or pd.isna(en):
#             continue
#         d0 = max(st.normalize(), start)
#         d1 = min(en.normalize(), end)
#         if d1 >= d0:
#             out.loc[d0:d1] = ph

#     out.name = "phase"
#     return out.fillna(fill_value)


DEFAULT_PHASE_MAP = {
    "moist": "moistening",
    "moisten": "moistening",
    "ripe": "ripening",
    "ripen": "ripening",
    "wet": "output",
    "melt": "output",
    "nan": "na",
    "none": "na",
}

ALLOWED = {"dry", "moistening", "ripening", "output", "na"}

def normalize_phase_series(s: pd.Series, mapping=None, fill_value="na") -> pd.Series:
    if mapping is None:
        mapping = DEFAULT_PHASE_MAP
    s2 = s.astype("object").apply(lambda v: str(v).lower().strip() if pd.notna(v) else np.nan)
    s2 = s2.replace(mapping).fillna(fill_value)
    return s2

def sm_3h_to_daily_majority(ds, phase_var="phase_name_3h") -> pd.Series:
    s = ds[phase_var].to_series()
    s.index = pd.to_datetime(s.index)
    daily = s.groupby(s.index.normalize()).agg(
        lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else np.nan
    )
    daily.name = "phase"
    return daily

def s1_intervals_to_daily_fill(ds, start, end,
                              start_var="interval_start",
                              end_var="interval_end",
                              phase_var="phase_name",
                              fill_value="na") -> pd.Series:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    idx = pd.date_range(start, end, freq="D")

    starts = pd.to_datetime(ds[start_var].values)
    ends   = pd.to_datetime(ds[end_var].values)
    phases = pd.Series(ds[phase_var].values).values

    order = np.argsort(starts.values)
    starts, ends, phases = starts[order], ends[order], phases[order]

    out = pd.Series(index=idx, dtype="object")
    for st, en, ph in zip(starts, ends, phases):
        if pd.isna(st) or pd.isna(en):
            continue
        d0 = max(st.normalize(), start)
        d1 = min(en.normalize(), end)
        if d1 >= d0:
            out.loc[d0:d1] = ph

    out.name = "phase"
    return out.fillna(fill_value)

def build_phases_by_year_daily(phases_by_year_xr, start_mmdd="01-01", end_mmdd="06-01"):
    """
    Convert phases by year from xarray Datasets or Sentinel dictionaries to daily pd.Series.
    
    phases_by_year_xr:
      {year: {"S1 m1": <xr.Dataset or dict>, "SM biased": <xr.Dataset>, "SM corrected": <xr.Dataset>, ...}}
      
    For S1 (Sentinel) methods: accepts Sentinel output dictionaries with 'qc_ok' and 'phases' keys.
    If qc_ok=False, the entire daily series is filled with "na".

    Returns:
      phases_by_year_daily:
      {year: {"S1 m1": <pd.Series daily>, ...}}
    """
    out = {}
    for yr, methods in phases_by_year_xr.items():
        start = f"{yr}-{start_mmdd}"
        end   = f"{yr}-{end_mmdd}"
        out[yr] = {}

        for name, obj in methods.items():
            if "S1" in name or "s1" in name:
                out[yr][name] = to_daily_phase_series_(obj, "s1", start, end)
            else:
                out[yr][name] = to_daily_phase_series_(obj, "sm", start, end)

    return out

def to_daily_phase_series_(obj, kind, start, end,
                          sm_phase_var="phase_name_3h",
                          s1_start_var="interval_start",
                          s1_end_var="interval_end",
                          s1_phase_var="phase_name",
                          mapping=None):
    """
    Convert xarray Dataset -> daily pd.Series of normalized phases.
    kind: "sm" or "s1"
    """
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")

    if kind == "sm":
        s = sm_3h_to_daily_majority(obj, phase_var=sm_phase_var).reindex(dates)
    elif kind == "s1":
        # Handle S1 output dicts with 'phases' and 'qc_ok' keys
        if isinstance(obj, dict) and "phases" in obj:
            if not obj.get("qc_ok", True):
                return pd.Series("na", index=dates, name="phase")
            obj = obj["phases"]
        s = s1_intervals_to_daily_fill(
            obj, start, end,
            start_var=s1_start_var, end_var=s1_end_var, phase_var=s1_phase_var
        ).reindex(dates)
    else:
        raise ValueError("kind must be 'sm' or 's1'")

    s = normalize_phase_series(s, mapping=mapping).fillna("na")

    # sanity: ensure no unknown labels
    u = set(pd.unique(s.astype(str).str.lower().str.strip()))
    unknown = sorted(u - ALLOWED)
    if unknown:
        raise ValueError(f"Unknown phase labels after normalization: {unknown}")

    return s

def daily_majority(s: pd.Series) -> pd.Series:
    """Resample a sub-daily (or irregular) Series to daily by majority vote."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    daily = s.groupby(s.index.normalize()).agg(
        lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else np.nan
    )
    daily.name = "phase"
    return daily

def normalize_phase_label(label) -> str:
    """Normalize a single phase label string (scalar version for Series.map)."""
    if pd.isna(label):
        return "na"
    low = str(label).lower().strip()
    return DEFAULT_PHASE_MAP.get(low, low)


def slice_year_window(s: pd.Series, yr: int, window_md=("01-01", "06-01")) -> pd.Series:
    """Slice a daily Series to [yr-mm-dd_start, yr-mm-dd_end]."""
    start = pd.to_datetime(f"{yr}-{window_md[0]}")
    end   = pd.to_datetime(f"{yr}-{window_md[1]}")
    return s.loc[start:end]


def phase_duration_days(s: pd.Series, phase: str) -> float:
    """Count the number of days where s equals phase."""
    n = (s == phase).sum()
    return float(n) if n > 0 else np.nan


def pairwise_agreement_matrix(series_dict, method_order, include_na=False):
    """
    Compute NxN pairwise agreement (%) and overlap-count matrices.
    series_dict: {method_name: pd.Series} of daily phase labels.
    Returns (A, N) where A[i,j] = % agreement, N[i,j] = # overlapping days.
    """
    n = len(method_order)
    A = np.full((n, n), np.nan)
    N = np.zeros((n, n), dtype=int)
    for i, mi in enumerate(method_order):
        for j, mj in enumerate(method_order):
            si = series_dict.get(mi, pd.Series(dtype="object"))
            sj = series_dict.get(mj, pd.Series(dtype="object"))
            if si.empty or sj.empty:
                continue
            aa, bb = si.align(sj, join="inner")
            aa = aa.astype("object")
            bb = bb.astype("object")
            mask = (~aa.isna()) & (~bb.isna())
            if not include_na:
                mask = mask & (aa != "na") & (bb != "na")
            count = int(mask.sum())
            if count == 0:
                continue
            A[i, j] = 100.0 * (aa[mask] == bb[mask]).mean()
            N[i, j] = count
    return A, N


def to_daily_phase_series_revamp(
    obj,
    sm_phase_vars=("phase_name_daily", "phase_name_3h", "phase_name"),
    s1_phase_var="phase_name",
    s1_start_coord="interval_start",
    s1_end_var="interval_end",
) -> pd.Series:
    """
    Supports:
      - Sentinel dictionary with 'qc_ok' and 'phases' keys (new format)
      - pd.Series
      - xr.DataArray with 'time'
      - xr.Dataset with:
          * 'time' dim + one of sm_phase_vars
          * OR interval style: dim 'interval' + coord interval_start + var interval_end + phase_name
    
    Returns: daily pd.Series of normalized labels.
    
    For Sentinel dictionaries: If qc_ok=False, returns NaN-filled series (rendered as "na").
    """
    # Handle Sentinel dictionary format with QC checking
    if isinstance(obj, dict) and ('qc_ok' in obj) and ('phases' in obj):
        if not obj.get('qc_ok', False):
            # QC failed: return empty series
            return pd.Series(dtype="object")
        # QC passed: extract phases dataset and process
        obj = obj['phases']
    
    # pandas
    if isinstance(obj, pd.Series):
        return daily_majority(obj)

    # xarray DataArray
    if isinstance(obj, xr.DataArray):
        if "time" not in obj.dims:
            raise ValueError(f"DataArray must have 'time' dimension. dims={obj.dims}")
        t = pd.to_datetime(obj["time"].values)
        s = pd.Series(obj.values, index=t)
        return daily_majority(s)

    # xarray Dataset
    if isinstance(obj, xr.Dataset):
        # interval-style Sentinel
        if ("interval" in obj.dims) and (s1_start_coord in obj.coords):
            return interval_dataset_to_daily(
                obj, phase_var=s1_phase_var, start_coord=s1_start_coord, end_var=s1_end_var
            )

        # time-style SnowModel (or already daily)
        if "time" in obj.dims:
            for v in sm_phase_vars:
                if v in obj.data_vars:
                    t = pd.to_datetime(obj["time"].values)
                    s = pd.Series(obj[v].values, index=t)
                    return daily_majority(s)
            raise ValueError(f"Dataset has 'time' but none of {sm_phase_vars} found. data_vars={list(obj.data_vars)}")

        raise ValueError(f"Unrecognized Dataset structure. dims={obj.dims}, coords={list(obj.coords)}, vars={list(obj.data_vars)}")

    raise TypeError(f"Unsupported phase container type: {type(obj)}")


# ============================================================
# MAIN: plot_summary_A_B_duration_output_doy
# ============================================================
def plot_summary_A_B_duration_output_doy(
    phases_by_year: dict,
    method_order=None,
    window_md=("01-01", "06-01"),
    duration_phases=("moistening", "ripening"),
    onset_phase="output",
    min_consecutive_days_onset=3,
    include_na_agreement=False,
    pooled_agreement=True,
    combine_duration_phases=False,
    combined_duration_label="Pre-runoff snow wetness",
    figsize=(12, 6),
    dpi=200,
    method_colors=None,
    method_markers=None,
    df_lys=None,
    lys_threshold=10,
    lys_min_consecutive_days=2,
    dry_duration=False,
    show_agreement: bool = True,
):
    """
    Panel A:
      - Moistening duration (days) across methods (spread bar + points)
      - Ripening duration (days) across methods (spread bar + points)
      - Output onset (DOY) across methods (spread bar + points)

    If combine_duration_phases=True:
      Panel A: Combined (moistening+ripening) duration
      Panel B: Output onset (DOY)
      Panel C: Pairwise agreement heatmap

    Panel B (or D if not combined):
      - Pairwise agreement heatmap (% days), pooled across years in the same window
        (or averaged per-year if pooled_agreement=False)

    Returns:
      fig, axes_tuple, metrics_df, (A, N), method_markers
    """
    # --- method order ---
    all_methods = sorted({m for yr in phases_by_year for m in phases_by_year[yr].keys()})
    if method_order is None:
        method_order = all_methods
    else:
        method_order = [m for m in method_order if m in all_methods]
        if len(method_order) == 0:
            raise ValueError(f"None of method_order found in phases_by_year. Available: {all_methods}")

    years = sorted(phases_by_year.keys())

    # --- style defaults ---
    if method_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        method_colors = {m: cycle[i % len(cycle)] for i, m in enumerate(method_order)}
    if method_markers is None:
        markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
        method_markers = {m: markers[i % len(markers)] for i, m in enumerate(method_order)}

    # --- compute per-year series (windowed) + metrics table ---
    metrics_rows = []
    per_year_windowed = {yr: {} for yr in years}

    for yr in years:
        for m in method_order:
            if m not in phases_by_year[yr]:
                continue

            s = to_daily_phase_series_revamp(phases_by_year[yr][m])
            s = s.map(normalize_phase_label)
            s = slice_year_window(s, yr, window_md=window_md)

            per_year_windowed[yr][m] = s

            metrics_rows.append(
                {
                    "year": yr,
                    "method": m,
                    f"dur_{duration_phases[0]}": phase_duration_days(s, duration_phases[0]),
                    f"dur_{duration_phases[1]}": phase_duration_days(s, duration_phases[1]),
                    "dur_dry": phase_duration_days(s, "dry"),
                    "onset_out_doy": onset_doy_consecutive(s, onset_phase, k=min_consecutive_days_onset),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)

    # Add combined duration column
    col_moist = f"dur_{duration_phases[0]}"
    col_ripe = f"dur_{duration_phases[1]}"
    # metrics_df["dur_combined"] = metrics_df[col_moist].fillna(0) + metrics_df[col_ripe].fillna(0)
    # metrics_df.loc[(metrics_df[col_moist].isna()) & (metrics_df[col_ripe].isna()), "dur_combined"] = np.nan
    metrics_df["dur_combined"] = metrics_df[col_moist].fillna(0) + metrics_df[col_ripe].fillna(0)
    metrics_df.loc[
        (metrics_df[col_moist].isna()) & (metrics_df[col_ripe].isna()),
        "dur_combined"
    ] = np.nan

    # Non-dry duration = moistening + ripening + output
    metrics_df["dur_nondry"] = (
        metrics_df[col_moist].fillna(0)
        + metrics_df[col_ripe].fillna(0)
        + metrics_df["onset_out_doy"].notna().astype(float) * 0  # placeholder keeps dtype float
    )

    # Better: compute directly from the daily series to include all output days
    nondry_lookup = []
    for _, row in metrics_df[["year", "method"]].iterrows():
        yr = row["year"]
        m = row["method"]
        s = per_year_windowed[yr][m]
        nd = float(((s == "moistening") | (s == "ripening") | (s == "output")).sum())
        nondry_lookup.append(nd if nd > 0 else np.nan)

    metrics_df["dur_nondry"] = nondry_lookup

    # --- agreement (pooled or averaged) ---
    if pooled_agreement:
        pooled = {m: [] for m in method_order}
        for yr in years:
            for m in method_order:
                if m in per_year_windowed[yr]:
                    pooled[m].append(per_year_windowed[yr][m])
        pooled = {m: pd.concat(pooled[m]).sort_index() if len(pooled[m]) else pd.Series(dtype="object") for m in pooled}
        A, N = pairwise_agreement_matrix(pooled, method_order=method_order, include_na=include_na_agreement)
    else:
        mats = []
        ns = []
        for yr in years:
            A_i, N_i = pairwise_agreement_matrix(per_year_windowed[yr], method_order=method_order, include_na=include_na_agreement)
            mats.append(A_i)
            ns.append(N_i.astype(float))
        A = np.nanmean(np.stack(mats), axis=0)
        N = np.nanmean(np.stack(ns), axis=0)

    # --- plot layout ---
    if not show_agreement:
        figsize = (figsize[0] * (1.5 / 2.5), figsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi)

    if combine_duration_phases:
        if show_agreement:
            gs_outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5, 1.0], wspace=0.25)
            axB = fig.add_subplot(gs_outer[0, 1])
        else:
            gs_outer = fig.add_gridspec(nrows=1, ncols=1)
            axB = None
        gs_left = gs_outer[0].subgridspec(nrows=2, ncols=2, width_ratios=[1.0, 0.08], wspace=0.0, hspace=0.45)
        axA1 = fig.add_subplot(gs_left[0, 0])
        axA2 = fig.add_subplot(gs_left[1, 0])
        axF1 = fig.add_subplot(gs_left[0, 1], sharey=axA1)
        axF2 = fig.add_subplot(gs_left[1, 1], sharey=axA2)
        failed_axes = [axF1, axF2]
        main_axes = [axA1, axA2]
    else:
        if show_agreement:
            gs_outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5, 1.0], wspace=0.25)
        else:
            gs_outer = fig.add_gridspec(nrows=1, ncols=1)

        # if dry_duration:
        #     # 5 rows: moistening, ripening, SPACER, dry, output onset
        #     # Row 2 is a spacer to create clear visual separation between (A,B) and (C,D) groups
        #     gs_left = gs_outer[0].subgridspec(nrows=5, ncols=2, width_ratios=[1.0, 0.08],
        #                                       height_ratios=[1.0, 1.0, 0.05, 1.0, 1.0], wspace=0.0, hspace=0.6)
        #     axA1 = fig.add_subplot(gs_left[0, 0])
        #     axA2 = fig.add_subplot(gs_left[1, 0], sharex=axA1)  # A and B share x-axis
        #     # Row 2 is spacer - no subplot
        #     axA3 = fig.add_subplot(gs_left[3, 0])  # dry duration (skip row 2)
        #     axA4 = fig.add_subplot(gs_left[4, 0])  # C and D share x-axis
        #     axF1 = fig.add_subplot(gs_left[0, 1])
        #     axF2 = fig.add_subplot(gs_left[1, 1])
        #     axF3 = fig.add_subplot(gs_left[3, 1])  # skip row 2
        #     axF4 = fig.add_subplot(gs_left[4, 1])  # skip row 2
        #     axB = fig.add_subplot(gs_outer[0, 1])
        #     failed_axes = [axF1, axF2, axF3, axF4]
        #     main_axes = [axA1, axA2, axA3, axA4]
        if dry_duration:
            # 4 rows: moistening, ripening, non-dry, sustained output onset
            gs_left = gs_outer[0].subgridspec(
                nrows=5, ncols=1,
                height_ratios=[1.0, 1.0, 0.001, 1.0, 1.0],
                hspace=0.6
            )
            axA1 = fig.add_subplot(gs_left[0, 0])
            axA2 = fig.add_subplot(gs_left[1, 0], sharex=axA1)
            axA3 = fig.add_subplot(gs_left[3, 0])
            axA4 = fig.add_subplot(gs_left[4, 0])
            axB = fig.add_subplot(gs_outer[0, 1]) if show_agreement else None

            main_axes = [axA1, axA2, axA3, axA4]
        else:
            # 3 rows: moistening, ripening, output onset
            gs_left = gs_outer[0].subgridspec(
                nrows=3, ncols=1,
                height_ratios=[1.0, 1.0, 1.2],
                hspace=0.45
            )
            axA1 = fig.add_subplot(gs_left[0, 0])
            axA2 = fig.add_subplot(gs_left[1, 0], sharex=axA1)
            axA3 = fig.add_subplot(gs_left[2, 0])
            axB = fig.add_subplot(gs_outer[0, 1]) if show_agreement else None

            main_axes = [axA1, axA2, axA3]

    # Common jitter settings
    n_methods = len(method_order)
    jitter_range = 0.5  # total spread around the year line

    def _plot_spread_points(ax, col, title, xlabel=None, ylabel=None):
        # spread bars per year (sized to span jitter range)
        for yr in years:
            sub = metrics_df[(metrics_df["year"] == yr) & (~metrics_df[col].isna())]
            if sub.empty:
                continue
            vals = sub[col].astype(float).values
            xmin, xmax = np.nanmin(vals), np.nanmax(vals)
            ax.barh(yr, width=xmax - xmin, left=xmin, height=jitter_range,
                    color="0.85", edgecolor="none", zorder=1)

        # method points with y-jitter to prevent overlapping
        for i, m in enumerate(method_order):
            sub = metrics_df[metrics_df["method"] == m]
            if sub.empty:
                continue
            # Compute y-offset: spread methods evenly from -jitter_range/2 to +jitter_range/2
            y_offset = (i / (n_methods - 1) - 0.5) * jitter_range if n_methods > 1 else 0
            ax.scatter(
                sub[col], sub["year"] + y_offset,
                s=60,
                marker=method_markers[m],
                color=method_colors[m],
                edgecolor="k",
                linewidth=0.5,
                zorder=3,
            )

        ax.set_title(title, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_yticks(years)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)

    def _plot_failed_qc(ax_fail, col, is_bottom=False):
        """Plot markers for S1 methods that failed QC (qc_ok=False) for each year."""
        # Filter to only S1 methods
        s1_methods = [m for m in method_order if "S1" in m or "s1" in m]
        n_s1 = len(s1_methods)
        if n_s1 == 0:
            ax_fail.set_visible(False)
            return

        # Plot failed S1 method markers side by side horizontally
        for i, m in enumerate(s1_methods):
            # Find years where this method failed QC
            failed_years = []
            for yr in years:
                if m in phases_by_year.get(yr, {}):
                    obj = phases_by_year[yr][m]
                    # Check if it's a dict with qc_ok key
                    if isinstance(obj, dict) and "qc_ok" in obj:
                        if not obj["qc_ok"]:
                            failed_years.append(yr)

            if not failed_years:
                continue

            # Spread S1 methods horizontally from -0.3 to 0.3
            x_offset = (i / (n_s1 - 1) - 0.5) * 0.6 if n_s1 > 1 else 0
            ax_fail.scatter(
                [x_offset] * len(failed_years), failed_years,
                s=60,
                marker=method_markers[m],
                color=method_colors[m],
                edgecolor="k",
                linewidth=0.5,
                zorder=3,
            )

        ax_fail.set_xlim(-0.5, 0.5)
        ax_fail.set_xticks([0])
        ax_fail.set_xticklabels(["Failed\nQC"] if is_bottom else [""], fontsize=10)
        ax_fail.tick_params(axis='y', labelleft=False)
        ax_fail.invert_yaxis()
        ax_fail.grid(False)

    if combine_duration_phases:
        # _plot_spread_points(axA1, "dur_combined",
        #                     f"A) {combined_duration_label} duration (days; Jan–Jun)")
        _plot_spread_points(axA3, "dur_nondry",
                                f"C) Non-dry duration (days; Jan–Jun)")
        _plot_spread_points(axA2, "onset_out_doy",
                            f"B) {onset_phase.capitalize()} onset (DOY; spread across methods)",
                            xlabel="Day of year (DOY)")
        # _plot_failed_qc(axF1, "dur_combined", is_bottom=False)
        # _plot_failed_qc(axF1, "dur_nondry", is_bottom=False)
        # _plot_failed_qc(axF2, "onset_out_doy", is_bottom=True)
        legend_ax = axA2
        heatmap_label = "C)"
        ax_output_onset = axA2
    else:
        axA1.set_xlim(0, 40)
        axA2.set_xlim(0, 40)
        # Hide x-tick labels on axA1 (shares x with axA2)
        axA1.tick_params(labelbottom=True)
        _plot_spread_points(axA1, col_moist,
                            f"A) {duration_phases[0].capitalize()} duration (Jan-Jun)",
                            ylabel="Water year")
        _plot_spread_points(axA2, col_ripe,
                            f"B) {duration_phases[1].capitalize()} duration (Jan-Jun)",
                            xlabel="Days",ylabel="Water year")
        # _plot_failed_qc(axF1, col_moist, is_bottom=False)
        # _plot_failed_qc(axF2, col_ripe, is_bottom=False)

        if dry_duration:
            # With dry_duration: A=moist, B=ripe, C=dry, D=output, E=heatmap
            # axA3.set_xlim(0, 175)
            # axA4.set_xlim(0, 175)
            axA3.set_xlim(0, 180)
            axA4.set_xlim(0, 180)
            # Hide x-tick labels on axA3 (shares x with axA4)
            axA3.tick_params(labelbottom=True)
            # _plot_spread_points(axA3, "dur_dry",
            #                     f"C) Dry duration (days; Jan–Jun)")
            _plot_spread_points(axA3, "dur_nondry",
                                f"C) Non-dry duration (Jan-Jun)",ylabel="Water year")
            _plot_spread_points(axA4, "onset_out_doy",
                                f"D) Sustained {onset_phase} (Jan-Jun)",
                                xlabel="Day of year (DOY)",ylabel="Water year")
            # _plot_failed_qc(axF3, "dur_dry", is_bottom=False)
            # _plot_failed_qc(axF3, "dur_nondry", is_bottom=False)
            # _plot_failed_qc(axF4, "onset_out_doy", is_bottom=True)
            legend_ax = axA4
            heatmap_label = "E)"
            ax_output_onset = axA4
        else:
            # Without dry_duration: A=moist, B=ripe, C=output, D=heatmap
            _plot_spread_points(axA3, "onset_out_doy",
                                f"C) Sustained {onset_phase} onset (Jan-Jun)",
                                xlabel="Day of year (DOY)")
            _plot_failed_qc(axF3, "onset_out_doy", is_bottom=True)
            axA3.set_xlim(0, 180)
            legend_ax = axA3
            heatmap_label = "D)"
            ax_output_onset = axA3

    # --- Lysimeter onset plotting ---
    if df_lys is not None:
        # tb4_south excluded permanently (persistent sensor failure: 0-3% of peak SWE annually)
        lys_cols_north = ['tb5_north', 'tb6_north', 'tb7_north', 'tb8_north']
        lys_cols_south = ['tb1_south', 'tb2_south', 'tb3_south']

        # Onset criterion: first day where daily meltwater sum >= lys_threshold (mm).
        # Default 0.6 mm/day matches the published "10+ tips/day" criterion.
        df_lys = df_lys.copy()
        df_lys['datetime'] = pd.to_datetime(df_lys['datetime'])
        df_lys_daily = df_lys.set_index('datetime').resample('D').sum().reset_index()

        def _get_lys_onset_doy_daily(df_daily, col, year, threshold, n_consecutive):
            """First DOY of n_consecutive days where daily meltwater sum >= threshold (mm).
            Search is restricted to the analysis window (window_md) to exclude
            out-of-season events (e.g. fall rain). Returns NaN if no run found."""
            win_start = pd.Timestamp(f'{year}-{window_md[0]}')
            win_end   = pd.Timestamp(f'{year}-{window_md[1]}')
            df_yr = df_daily[
                (df_daily['datetime'] >= win_start) &
                (df_daily['datetime'] <= win_end)
            ].copy()
            above = df_yr[col] >= threshold
            run = 0
            for idx, val in zip(df_yr['datetime'], above):
                run = run + 1 if val else 0
                if run >= n_consecutive:
                    onset_date = idx - pd.Timedelta(days=n_consecutive - 1)
                    return onset_date.dayofyear
            return np.nan

        lys_onset_rows = []
        for yr in years:
            for grp, cols in [('Lys-1', lys_cols_north), ('Lys-2', lys_cols_south)]:
                onsets = [_get_lys_onset_doy_daily(df_lys_daily, col, yr, lys_threshold, lys_min_consecutive_days)
                          for col in cols]
                onsets = [x for x in onsets if not np.isnan(x)]
                if onsets:
                    lys_onset_rows.append({
                        'year':       yr,
                        'group':      grp,
                        'median_doy': np.median(onsets),
                        'min_doy':    np.min(onsets),
                        'max_doy':    np.max(onsets),
                    })

        df_lys_onset = pd.DataFrame(lys_onset_rows)

        lys_colors  = {'Lys-1': 'darkgreen', 'Lys-2': 'limegreen'}
        lys_markers = {'Lys-1': 'H', 'Lys-2': 'h'}

        for grp in ['Lys-1', 'Lys-2']:
            sub = df_lys_onset[df_lys_onset['group'] == grp]
            if sub.empty:
                continue
            color = lys_colors[grp]

            # Range bar (min–max across tubes), same color as median marker
            for _, row in sub.iterrows():
                ax_output_onset.barh(
                    row['year'],
                    width=row['max_doy'] - row['min_doy'],
                    left=row['min_doy'],
                    height=jitter_range,
                    color=color,
                    alpha=0.3,
                    edgecolor='none',
                    zorder=2,
                )

            # Median marker
            ax_output_onset.scatter(
                sub['median_doy'], sub['year'],
                s=70,
                marker=lys_markers[grp],
                color=color,
                edgecolor='k',
                linewidth=0.5,
                zorder=4,
                label=grp,
            )


    # figure-level legend (clean)
    handles = [
        Line2D([0], [0],
               marker=method_markers[m],
               linestyle="None",
               markerfacecolor=method_colors[m],
               markeredgecolor="k",
               markersize=7,
               label=m)
        for m in method_order
    ]

    # Add lysimeter handles if df_lys was provided
    if df_lys is not None:
        lys_colors = {'Lysimeter (grouping 1)': 'darkgreen', 'Lysimeter (grouping 2)': 'limegreen'}
        lys_markers = {'Lysimeter (grouping 1)': 'H', 'Lysimeter (grouping 2)': 'h'}
        for grp in ['Lysimeter (grouping 1)', 'Lysimeter (grouping 2)']:
            handles.append(
                Line2D([0], [0],
                       marker=lys_markers[grp],
                       linestyle="None",
                       markerfacecolor=lys_colors[grp],
                       markeredgecolor="k",
                       markersize=7,
                       label=grp)
            )

    axA_bottom = legend_ax
    # axA_bottom.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
    #            ncol=len(method_order), frameon=True, title="Method",
    #            fontsize=8, title_fontsize=9, handletextpad=0.4, columnspacing=1.0)
    axA_bottom.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
               ncol=4, frameon=True, title="Method",
               fontsize=8, title_fontsize=9, handletextpad=0.4, columnspacing=1.0)

    # Panel E: agreement heatmap (optional)
    if show_agreement and axB is not None:
        im = axB.imshow(A, vmin=40, vmax=100, aspect="auto")
        axB.set_title(f"{heatmap_label} Pairwise agreement (% days; pooled Jan–Jun)", fontweight="bold", fontsize=9)
        axB.set_xticks(np.arange(len(method_order)))
        axB.set_yticks(np.arange(len(method_order)))
        axB.set_xticklabels(method_order, rotation=30, ha="right", fontsize=7)
        axB.set_yticklabels(method_order, rotation=45, ha="right", fontsize=7)
        for i in range(len(method_order)):
            for j in range(len(method_order)):
                if np.isfinite(A[i, j]):
                    axB.text(j, i, f"{A[i,j]:.0f}\n(n={int(N[i,j])})", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
        cbar.set_label("Agreement (%)")

    plt.tight_layout()
    if combine_duration_phases:
        return fig, (axA1, axA2, axB), metrics_df, (A, N), method_markers
    elif dry_duration:
        return fig, (axA1, axA2, axA3, axA4, axB), metrics_df, (A, N), method_markers
    else:
        return fig, (axA1, axA2, axA3, axB), metrics_df, (A, N), method_markers


# ============================================================
# Interval dataset → daily
# ============================================================
def interval_dataset_to_daily(ds, phase_var="phase_name",
                              start_coord="interval_start",
                              end_var="interval_end",
                              fill_value="na") -> pd.Series:
    """Convert an interval-style xr.Dataset to a daily pd.Series."""
    starts = pd.to_datetime(ds[start_coord].values)
    ends   = pd.to_datetime(ds[end_var].values)
    idx = pd.date_range(starts.min().normalize(), ends.max().normalize(), freq="D")

    order = np.argsort(starts.values)
    starts, ends = starts[order], ends[order]
    phases = pd.Series(ds[phase_var].values).values[order]

    out = pd.Series(index=idx, dtype="object")
    for st, en, ph in zip(starts, ends, phases):
        if pd.isna(st) or pd.isna(en):
            continue
        d0 = st.normalize()
        d1 = en.normalize()
        if d1 >= d0:
            out.loc[d0:d1] = ph
    out.name = "phase"
    return out.fillna(fill_value)

def onset_doy_consecutive(s: pd.Series, phase: str, k: int = 3) -> float:
    """Return the DOY of the first day of k consecutive days matching phase."""
    is_phase = (s == phase).astype(int)
    hits = is_phase.rolling(k, min_periods=k).sum()
    cand = hits[hits == k]
    if cand.empty:
        return np.nan
    first = cand.index[0] - pd.Timedelta(days=k - 1)
    return float(first.dayofyear)

# ============================================================
# NEW FOCUSED FUNCTIONS
# ============================================================

def compute_phase_metrics(
    phases_by_year: dict,
    method_order=None,
    window_md=("01-01", "06-01"),
    duration_phases=("moistening", "ripening"),
    onset_phase="output",
    min_consecutive_days_onset=3,
    include_na_agreement=False,
    pooled_agreement=True,
    method_colors=None,
    method_markers=None,
):
    """
    Compute per-year phase metrics and pairwise agreement.
    Returns a dict that can be passed directly to plot_duration_panels()
    and plot_onset_panels().
    """
    all_methods = sorted({m for yr in phases_by_year for m in phases_by_year[yr].keys()})
    if method_order is None:
        method_order = all_methods
    else:
        method_order = [m for m in method_order if m in all_methods]

    years = sorted(phases_by_year.keys())

    if method_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        method_colors = {m: cycle[i % len(cycle)] for i, m in enumerate(method_order)}
    if method_markers is None:
        markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
        method_markers = {m: markers[i % len(markers)] for i, m in enumerate(method_order)}

    col_moist = f"dur_{duration_phases[0]}"
    col_ripe  = f"dur_{duration_phases[1]}"

    metrics_rows = []
    per_year_windowed = {yr: {} for yr in years}

    for yr in years:
        for m in method_order:
            if m not in phases_by_year[yr]:
                continue
            s = to_daily_phase_series_revamp(phases_by_year[yr][m])
            s = s.map(normalize_phase_label)
            s = slice_year_window(s, yr, window_md=window_md)
            per_year_windowed[yr][m] = s
            metrics_rows.append({
                "year":    yr,
                "method":  m,
                col_moist: phase_duration_days(s, duration_phases[0]),
                col_ripe:  phase_duration_days(s, duration_phases[1]),
                "dur_dry": phase_duration_days(s, "dry"),
                "onset_out_doy": onset_doy_consecutive(s, onset_phase, k=min_consecutive_days_onset),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df[col_moist] = metrics_df[col_moist]
    metrics_df[col_ripe]  = metrics_df[col_ripe]

    # Non-dry duration computed directly from daily series
    nondry = []
    for _, row in metrics_df[["year", "method"]].iterrows():
        s = per_year_windowed[row["year"]][row["method"]]
        nd = float(((s == "moistening") | (s == "ripening") | (s == "output")).sum())
        nondry.append(nd if nd > 0 else np.nan)
    metrics_df["dur_nondry"] = nondry

    # Agreement matrix
    if pooled_agreement:
        pooled = {m: [] for m in method_order}
        for yr in years:
            for m in method_order:
                if m in per_year_windowed[yr]:
                    pooled[m].append(per_year_windowed[yr][m])
        pooled = {m: pd.concat(pooled[m]).sort_index() if pooled[m] else pd.Series(dtype="object")
                  for m in pooled}
        A, N = pairwise_agreement_matrix(pooled, method_order=method_order, include_na=include_na_agreement)
    else:
        mats, ns = [], []
        for yr in years:
            A_i, N_i = pairwise_agreement_matrix(per_year_windowed[yr], method_order=method_order,
                                                  include_na=include_na_agreement)
            mats.append(A_i); ns.append(N_i.astype(float))
        A = np.nanmean(np.stack(mats), axis=0)
        N = np.nanmean(np.stack(ns),   axis=0)

    return dict(
        metrics_df=metrics_df,
        method_order=method_order,
        method_colors=method_colors,
        method_markers=method_markers,
        years=years,
        A=A, N=N,
        duration_phases=duration_phases,
        col_moist=col_moist,
        col_ripe=col_ripe,
        window_md=window_md,
    )


def _draw_spread_points(ax, col, metrics_df, years, method_order,
                        method_colors, method_markers,
                        title, xlabel=None, ylabel=None, jitter_range=0.5):
    """Gray spread bar + per-method scatter points for one panel."""
    n_methods = len(method_order)
    for yr in years:
        sub = metrics_df[(metrics_df["year"] == yr) & metrics_df[col].notna()]
        if sub.empty:
            continue
        vals = sub[col].astype(float).values
        ax.barh(yr, width=np.nanmax(vals) - np.nanmin(vals), left=np.nanmin(vals),
                height=jitter_range, color="0.85", edgecolor="none", zorder=1)

    for i, m in enumerate(method_order):
        sub = metrics_df[metrics_df["method"] == m]
        if sub.empty:
            continue
        y_off = (i / (n_methods - 1) - 0.5) * jitter_range if n_methods > 1 else 0
        ax.scatter(sub[col], sub["year"] + y_off, s=60,
                   marker=method_markers[m], color=method_colors[m],
                   edgecolor="k", linewidth=0.5, zorder=3)

    ax.set_title(title, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_yticks(years)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle="-", color="gray", alpha=0.2, linewidth=0.6)


def _build_legend_handles(method_order, method_colors, method_markers,
                          lys_colors=None, lys_markers=None):
    handles = [
        Line2D([0], [0], marker=method_markers[m], linestyle="None",
               markerfacecolor=method_colors[m], markeredgecolor="k",
               markersize=7, label=m)
        for m in method_order
    ]
    if lys_colors and lys_markers:
        for grp in lys_colors:
            handles.append(
                Line2D([0], [0], marker=lys_markers[grp], linestyle="None",
                       markerfacecolor=lys_colors[grp], markeredgecolor="k",
                       markersize=7, label=grp)
            )
    return handles


def plot_duration_panels(
    metrics: dict,
    plotInsetTitle=False,
    figsize=(8, 5),
    dpi=200,
):
    """
    Figure 1: A) Moistening duration  B) Ripening duration.
    Parameters
    ----------
    metrics : dict returned by compute_phase_metrics()
    plotInsetTitle : if True, titles are placed as text in the top-left corner
                     of each subplot instead of as axis titles.
    """
    md  = metrics["metrics_df"]
    mo  = metrics["method_order"]
    mc  = metrics["method_colors"]
    mm  = metrics["method_markers"]
    yrs = metrics["years"]

    title_A = f"a) {metrics['duration_phases'][0].capitalize()} duration\n(Jan–Jun)"
    title_B = f"b) {metrics['duration_phases'][1].capitalize()} duration\n(Jan–Jun)"

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs  = fig.add_gridspec(nrows=2, ncols=1, hspace=0.15)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)

    _draw_spread_points(axA, metrics["col_moist"], md, yrs, mo, mc, mm,
                        title_A, ylabel="Water year")
    _draw_spread_points(axB, metrics["col_ripe"],  md, yrs, mo, mc, mm,
                        title_B, xlabel="Days", ylabel="Water year")
    axA.set_xlim(0, 40)
    axA.tick_params(labelbottom=False)   # B carries the shared x-axis label

    if plotInsetTitle:
        for ax, title in [(axA, title_A), (axB, title_B)]:
            ax.set_title("")
            ax.text(0.99, 0.97, title, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="right")

    from matplotlib.patches import Patch
    handles = _build_legend_handles(mo, mc, mm)
    handles.append(Patch(facecolor="0.85", edgecolor="none", label="Method spread"))
    axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
               ncol=4, frameon=True, title="Method",
               fontsize=8, title_fontsize=9, handletextpad=0.4, columnspacing=1.0)

    plt.tight_layout()
    return fig, (axA, axB)


def plot_onset_panels(
    metrics: dict,
    df_lys=None,
    lys_threshold=0.6,
    lys_min_consecutive_days=2,
    swe_ds=None,
    plotInsetTitle=False,
    plotDOY=False,
    dualPanel=True,
    figsize=(8, 9),
    dpi=200,
):
    """
    Figure 2: C) Non-dry duration  D) Sustained output onset DOY.

    Panel D layout — each water year occupies two horizontal rows:
      Row 1: model/S1 scatter points + gray spread bar
      Row 2: lysimeter median markers + coloured range bars
    Year blocks separated by dashed horizontal lines.
    Peak SWE DOY (from swe_ds) plotted as a vertical dashed black line
    spanning both rows of each year.

    Returns fig, (axC, axD), onset_table
      onset_table : DataFrame with columns [year, measurement, doy, n_obs]
    """
    from matplotlib.patches import Patch

    md        = metrics["metrics_df"]
    mo        = metrics["method_order"]
    mc        = metrics["method_colors"]
    mm        = metrics["method_markers"]
    yrs       = metrics["years"]
    window_md = metrics["window_md"]
    n_methods = len(mo)

    lys_colors  = {"Lys-1": "darkgreen", "Lys-2": "limegreen"}
    lys_markers = {"Lys-1": "H",         "Lys-2": "h"}
    lys_cols_north = ["tb5_north", "tb6_north", "tb7_north", "tb8_north"]
    lys_cols_south = ["tb1_south", "tb2_south", "tb3_south"]

    # ── Peak SWE DOY per year ────────────────────────────────────────────────
    peak_swe_doys = {}
    if swe_ds is not None:
        for yr in yrs:
            swe_yr = swe_ds.sel(time=slice(f"{yr-1}-10-01", f"{yr}-09-30"))
            vals   = swe_yr["swed"].values.ravel()
            if np.all(np.isnan(vals)):
                peak_swe_doys[yr] = np.nan
            else:
                peak_swe_doys[yr] = int(
                    pd.Timestamp(swe_yr.time.values[np.nanargmax(vals)]).dayofyear
                )

    # ── Lysimeter onset computation ──────────────────────────────────────────
    df_lys_onset   = pd.DataFrame()
    lys_tube_onsets = {}
    df_lys_daily   = None

    if df_lys is not None:
        df_lys = df_lys.copy()
        df_lys["datetime"] = pd.to_datetime(df_lys["datetime"])
        df_lys_daily = df_lys.set_index("datetime").resample("D").sum().reset_index()

        def _lys_onset(df_daily, col, year):
            win_start = pd.Timestamp(f"{year}-{window_md[0]}")
            win_end   = pd.Timestamp(f"{year}-{window_md[1]}")
            df_yr = df_daily[(df_daily["datetime"] >= win_start) &
                              (df_daily["datetime"] <= win_end)].copy()
            run = 0
            for dt, val in zip(df_yr["datetime"], df_yr[col] >= lys_threshold):
                run = run + 1 if val else 0
                if run >= lys_min_consecutive_days:
                    return (dt - pd.Timedelta(days=lys_min_consecutive_days - 1)).dayofyear
            return np.nan

        lys_rows = []
        for yr in yrs:
            for grp, cols in [("Lys-1", lys_cols_north), ("Lys-2", lys_cols_south)]:
                onsets = [_lys_onset(df_lys_daily, c, yr) for c in cols]
                valid  = [x for x in onsets if not np.isnan(x)]
                lys_tube_onsets[(yr, grp)] = valid
                if valid:
                    lys_rows.append({"year": yr, "group": grp,
                                     "mean_doy": np.mean(valid),
                                     "min_doy":  np.min(valid),
                                     "max_doy":  np.max(valid),
                                     "n":        len(valid)})
        df_lys_onset = pd.DataFrame(lys_rows)

    # ── Y-axis block layout for panel D ─────────────────────────────────────
    block   = 2.5
    jr_mod  = 0.45
    lys_off = 0.18

    y_m   = {yr: i * block       for i, yr in enumerate(yrs)}
    y_l   = {yr: i * block + 1.0 for i, yr in enumerate(yrs)}
    y_mid = {yr: i * block + 0.5 for i, yr in enumerate(yrs)}

    # ── Figure / axes ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if dualPanel:
        gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 2], hspace=0.1)
        axC = fig.add_subplot(gs[0])
        axD = fig.add_subplot(gs[1], sharex=axC)
    else:
        gs  = fig.add_gridspec(nrows=1, ncols=1)
        axC = None
        axD = fig.add_subplot(gs[0])

    # ── Panel C: non-dry duration ────────────────────────────────────────────
    if dualPanel:
        _draw_spread_points(axC, "dur_nondry", md, yrs, mo, mc, mm,
                            "A) Wet duration\n(Jan–Jun)",
                            ylabel="Water year")
        axC.tick_params(labelbottom=False)   # suppress x-ticks; D carries the shared label
        axC.set_xlim(0, 190)
        if plotInsetTitle:
            axC.set_title("")
            axC.text(0.01, 0.97, "A) Wet duration\n(Jan–Jun)",
                     transform=axC.transAxes, fontsize=9, fontweight="bold",
                     va="top", ha="left")

    # ── Panel D: two-row-per-year onset ──────────────────────────────────────
    for yr in yrs:
        ym = y_m[yr]

        # Gray spread bar for methods range
        sub_yr = md[(md["year"] == yr) & md["onset_out_doy"].notna()]
        if not sub_yr.empty:
            vals = sub_yr["onset_out_doy"].astype(float).values
            axD.barh(ym, width=np.nanmax(vals) - np.nanmin(vals),
                     left=np.nanmin(vals), height=jr_mod,
                     color="0.85", edgecolor="none", zorder=1)

        # Method scatter points
        for i, m in enumerate(mo):
            row_m = md[(md["year"] == yr) & (md["method"] == m)]
            if row_m.empty or row_m["onset_out_doy"].isna().all():
                continue
            y_off = (i / (n_methods - 1) - 0.5) * jr_mod if n_methods > 1 else 0
            axD.scatter(row_m["onset_out_doy"].values, [ym + y_off],
                        s=60, marker=mm[m], color=mc[m],
                        edgecolor="k", linewidth=0.5, zorder=3)

        # Lysimeter range bars and median markers
        if not df_lys_onset.empty:
            for grp, g_off in [("Lys-1", -lys_off), ("Lys-2", +lys_off)]:
                row_l = df_lys_onset[(df_lys_onset["year"] == yr) &
                                      (df_lys_onset["group"] == grp)]
                if row_l.empty:
                    continue
                r     = row_l.iloc[0]
                color = lys_colors[grp]
                yl    = y_l[yr] + g_off
                axD.barh(yl, width=r["max_doy"] - r["min_doy"],
                         left=r["min_doy"], height=0.28,
                         color=color, alpha=0.3, edgecolor="none", zorder=2)
                axD.scatter(r["mean_doy"], yl,
                            s=70, marker=lys_markers[grp], color=color,
                            edgecolor="k", linewidth=0.5, zorder=4)

        # Peak SWE vertical dashed line spanning both rows of this year
        peak_doy = peak_swe_doys.get(yr, np.nan)
        if not np.isnan(peak_doy):
            axD.vlines(peak_doy,
                       ymin=y_m[yr] - jr_mod / 2 - 0.05,
                       ymax=y_l[yr] + lys_off + 0.22,
                       colors="black", linestyles="--",
                       linewidth=1.5, zorder=5)

    # Horizontal dividing lines between year blocks
    for i in range(len(yrs) - 1):
        axD.axhline(i * block + 1.75, color="gray", linewidth=0.8,
                    linestyle="-", alpha=0.5, zorder=0)

    # Y-axis formatting
    axD.set_yticks(list(y_mid.values()))
    axD.set_yticklabels([str(yr) for yr in yrs])
    axD.set_ylim(-0.6, (len(yrs) - 1) * block + 1.6)
    axD.invert_yaxis()
    axD.set_ylabel("Water year")
    axD.grid(axis="x", linestyle="--", alpha=0.4)
    axD.grid(axis="y", visible=False)

    if plotDOY:
        from datetime import date
        ref_year = 2019  # non-leap reference year for DOY -> month/day conversion
        month_doys   = [date(ref_year, m, 1).timetuple().tm_yday for m in range(1, 9)]
        month_labels = [date(ref_year, m, 1).strftime("%b 1")    for m in range(1, 9)]
        axD.set_xticks(month_doys)
        axD.set_xticklabels(month_labels, rotation=45, ha="right")
        if dualPanel:
            axC.set_xticks(month_doys)
        axD.set_xlabel("Date")
    else:
        axD.set_xlabel("Day of year (DOY)")

    title_D = ("B) Sustained output onset\n(Jan–Jun)" if dualPanel
               else "Sustained output onset (Jan-Jun)")
    if plotInsetTitle:
        axD.text(0.01, 0.97, title_D,
                 transform=axD.transAxes, fontsize=9, fontweight="bold",
                 va="top", ha="left", zorder=6)
    else:
        axD.set_title(title_D, fontweight="bold")

    # ── Two legends ──────────────────────────────────────────────────────────
    handles_method = [
        Line2D([0], [0], marker=mm[m], linestyle="None",
               markerfacecolor=mc[m], markeredgecolor="k",
               markersize=7, label=m)
        for m in mo
    ]
    handles_method.append(Patch(facecolor="0.85", edgecolor="none", label="Method spread"))

    handles_obs = [
        Line2D([0], [0], marker="H", linestyle="None",
               markerfacecolor="darkgreen", markeredgecolor="k",
               markersize=7, label="Lys mean grp 1"),
        Line2D([0], [0], marker="h", linestyle="None",
               markerfacecolor="limegreen", markeredgecolor="k",
               markersize=7, label="Lys mean grp 2"),
    ]
    if swe_ds is not None:
        handles_obs.append(
            Line2D([0], [0], color="black", linestyle="--",
                   linewidth=1.5, label="Peak SWE DOY")
        )


    handles_obs.append(Patch(facecolor="darkgreen", alpha=0.3, edgecolor="none", label="Lys spread grp 1"))
    handles_obs.append(Patch(facecolor="limegreen", alpha=0.3, edgecolor="none", label="Lys spread grp 2"))


    leg1 = axD.legend(handles=handles_method, title="Method",
                      loc="upper left", bbox_to_anchor=(-0.03, -0.14),
                      frameon=True, fontsize=8, title_fontsize=9,
                      ncol=3, handletextpad=0.4, columnspacing=1.0)
    axD.add_artist(leg1)
    axD.legend(handles=handles_obs, title="Observations",
               loc="upper right", bbox_to_anchor=(1.0, -0.14),
               frameon=True, fontsize=8, title_fontsize=9,
               ncol=2, handletextpad=0.4, columnspacing=1.0)

    # ── Onset table ──────────────────────────────────────────────────────────
    table_rows = []
    for yr in yrs:
        for m in mo:
            sub = md[(md["year"] == yr) & (md["method"] == m)]
            doy_val = float(sub["onset_out_doy"].iloc[0]) if not sub.empty else np.nan
            table_rows.append({"year": yr, "measurement": m,
                                "doy": doy_val, "n_obs": 1})
        for grp in ["Lys-1", "Lys-2"]:
            valid = lys_tube_onsets.get((yr, grp), [])
            n = len(valid)
            for stat, val in [("mean",   np.mean(valid)   if n else np.nan),
                               ("median", np.median(valid) if n else np.nan),
                               ("min",    np.min(valid)    if n else np.nan),
                               ("max",    np.max(valid)    if n else np.nan)]:
                table_rows.append({"year": yr,
                                   "measurement": f"{grp} {stat}",
                                   "doy": val, "n_obs": n})
        if swe_ds is not None:
            table_rows.append({"year": yr, "measurement": "Peak SWE DOY",
                                "doy": peak_swe_doys.get(yr, np.nan), "n_obs": 1})

    onset_table = pd.DataFrame(table_rows)

    fig.subplots_adjust(bottom=0.22, hspace=0.1)
    return fig, (axC, axD), onset_table
