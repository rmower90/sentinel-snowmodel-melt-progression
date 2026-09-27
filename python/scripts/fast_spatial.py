"""
fast_spatial.py — Efficient spatial melt-phase classification.

Loads all SnowModel variables for one water year into pre-allocated numpy
arrays (one-time I/O cost), then classifies pixels in parallel using
multiprocessing.Pool with fork-based shared memory.

Usage from notebook:
    import fast_spatial
    spatial_data = fast_spatial.load_wy_to_memory(sm_dir, wy, snowmodel_dict)
    pt_data = fast_spatial.extract_point_datasets(spatial_data, pixel_indices)
    result  = fast_spatial.classify_spatial_parallel(spatial_data, ...)
"""

import os
import time as _time_mod
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import melt_phases


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wy_to_memory(sm_output_dir, wy, snowmodel_dict):
    """
    Load all four SM variables for one water year into pre-allocated numpy
    arrays, reading files one-by-one to avoid HDF file-handle exhaustion.

    Parameters
    ----------
    sm_output_dir  : str  — root SM output dir (e.g. SM_BIASED_ROOT)
    wy             : int  — water year (e.g. 2017)
    snowmodel_dict : dict — maps wy → directory name (e.g. {2017: 'wy_2017'})

    Returns
    -------
    dict with keys 'melt', 'tsfc', 'roff', 'swed' (numpy float32 arrays)
    and 'time' (datetime64[ns]). Multilayer fields are NaN-masked where
    tsfc < -9998.
    """
    nc_dir = f'{sm_output_dir}{snowmodel_dict[wy]}/netcdf/'
    t_start = np.datetime64(f'{wy - 1}-10-01')
    t_end = np.datetime64(f'{wy}-09-01')

    VAR_MAP = [
        ('multilayer_liq',  'sm_multilayer_liq_',  'melt'),
        ('multilayer_temp', 'sm_multilayer_temp_', 'tsfc'),
        ('roff',            'sm_roff_',             'roff'),
        ('swed',            'sm_swed_',             'swed'),
    ]

    data = {}
    t0_all = _time_mod.time()
    times_arr = None

    for label, file_prefix, data_var in VAR_MAP:
        files = sorted(
            nc_dir + f
            for f in os.listdir(nc_dir)
            if f.startswith(file_prefix) and f.endswith('.nc')
        )
        nt = len(files)

        t0 = _time_mod.time()
        print(f'  [{label}] {nt} files ...', end=' ', flush=True)

        # Probe first file for shape (excluding time dim)
        with xr.open_dataset(files[0]) as ds0:
            spatial_shape = ds0[data_var].shape[1:]

        arr = np.empty((nt,) + spatial_shape, dtype=np.float32)
        if times_arr is None:
            times_arr = np.empty(nt, dtype='datetime64[ns]')

        for i, fpath in enumerate(files):
            with xr.open_dataset(fpath) as ds:
                arr[i] = ds[data_var].values[0]
                if data_var == 'melt':  # capture time once
                    times_arr[i] = ds.time.values[0]

        data[data_var] = arr
        print(f'{_time_mod.time() - t0:.1f}s  shape={arr.shape}')

    # Slice to water year
    mask = (times_arr >= t_start) & (times_arr <= t_end)
    for key in ('melt', 'tsfc', 'roff', 'swed'):
        data[key] = data[key][mask]
    data['time'] = times_arr[mask]

    # Apply fill-value mask to multilayer fields
    ml_mask = data['tsfc'] >= -9998.0
    data['melt'] = np.where(ml_mask, data['melt'], np.nan)
    data['tsfc'] = np.where(ml_mask, data['tsfc'], np.nan)

    nt_final = len(data['time'])
    mem_gb = sum(data[k].nbytes for k in ('melt', 'tsfc', 'roff', 'swed')) / 1e9
    print(f'  Loaded: {nt_final} timesteps, {mem_gb:.1f} GB, '
          f'{_time_mod.time() - t0_all:.1f}s wall time')

    return data


def extract_point_datasets(spatial_data, pixel_indices):
    """
    Extract point xr.Datasets from pre-loaded spatial arrays.

    Parameters
    ----------
    spatial_data  : dict from load_wy_to_memory()
    pixel_indices : dict[str, (int, int)]  e.g. {'TUM': (iy, ix), ...}

    Returns
    -------
    dict[str, xr.Dataset]  — each has tsfc, melt, swed, roff ready for
    classify_snowmodel_melt_phases_3h_full().
    """
    time_np = spatial_data['time']
    pt = {}
    for site, (iy, ix) in pixel_indices.items():
        pt[site] = xr.Dataset({
            'tsfc': xr.DataArray(
                spatial_data['tsfc'][:, :, iy, ix],
                dims=['time', 'layer'], coords={'time': time_np}),
            'melt': xr.DataArray(
                spatial_data['melt'][:, :, iy, ix],
                dims=['time', 'layer'], coords={'time': time_np}),
            'swed': xr.DataArray(
                spatial_data['swed'][:, iy, ix],
                dims=['time'], coords={'time': time_np}),
            'roff': xr.DataArray(
                spatial_data['roff'][:, iy, ix],
                dims=['time'], coords={'time': time_np}),
        })
    return pt


# ---------------------------------------------------------------------------
# Parallel classification — module-level state for fork-inherited workers
# ---------------------------------------------------------------------------

_MP_DATA = None
_MP_THRESH = None
_MP_METHOD = 'original'   # 'original' or 'revamp'
_MP_KWARGS = {}           # extra kwargs forwarded to classifier


def _mp_classify_pixel(pixel):
    """
    Worker: classify one (south_north, east_west) pixel.

    Accesses _MP_DATA / _MP_THRESH / _MP_METHOD / _MP_KWARGS via
    fork-inherited module globals (copy-on-write, zero extra memory).
    """
    iy, ix = pixel
    d = _MP_DATA

    swed_px = d['swed'][:, iy, ix]
    # Skip no-data or trivially snow-free pixels
    if np.all(np.isnan(swed_px)) or np.nanmax(swed_px) < 0.01:
        return None

    time_np = d['time']
    ds_pt = xr.Dataset({
        'tsfc': xr.DataArray(d['tsfc'][:, :, iy, ix],
                             dims=['time', 'layer'],
                             coords={'time': time_np}),
        'melt': xr.DataArray(d['melt'][:, :, iy, ix],
                             dims=['time', 'layer'],
                             coords={'time': time_np}),
        'swed': xr.DataArray(swed_px,
                             dims=['time'], coords={'time': time_np}),
        'roff': xr.DataArray(d['roff'][:, iy, ix],
                             dims=['time'], coords={'time': time_np}),
    })

    if _MP_METHOD == 'revamp':
        result = melt_phases.classify_snowmodel_melt_phases_3h_full_revamp(
            ds_pt, **_MP_KWARGS,
        )
    else:
        result = melt_phases.classify_snowmodel_melt_phases_3h_full(
            ds_pt, water_thresh=_MP_THRESH,
        )

    code = result['phase_code_3h'].values.astype(np.int8)
    out_mask = code == 3
    out_start = (result['time'].values[out_mask][0]
                 if out_mask.any()
                 else np.datetime64('NaT'))

    return {
        'iy': iy,
        'ix': ix,
        'code': code,
        'time': result['time'].values,
        'out_start': out_start,
    }


def classify_spatial_parallel(
    spatial_data,
    y_vals,
    x_vals,
    water_thresh=0.0035,
    start_date=None,
    end_date=None,
    stride=1,
    n_workers=8,
    method='original',
    **classifier_kwargs,
):
    """
    Classify melt phases for every grid pixel using multiprocessing.Pool.

    Parameters
    ----------
    spatial_data   : dict from load_wy_to_memory() — already in RAM
    y_vals, x_vals : 1-D projected coordinate arrays
    water_thresh   : internal-liquid threshold for original classifier
    start_date     : ISO date string for daily output start
    end_date       : ISO date string for daily output end
    stride         : subsample every Nth pixel (1 = full resolution)
    n_workers      : parallel workers (0 or 1 = serial fallback)
    method         : 'original' or 'revamp' — selects classifier function
    **classifier_kwargs : extra keyword args forwarded to the revamp classifier
                          (e.g. water_thresh_surface, water_thresh_internal,
                          min_runoff_rate, output_merge_dry_hours, ripen_gap_days)

    Returns
    -------
    xr.Dataset with (time, south_north, east_west) dims matching the
    original run_sm_spatial() output structure.
    """
    import fast_spatial  # ensure module reference for pickle
    fast_spatial._MP_DATA = spatial_data
    fast_spatial._MP_THRESH = water_thresh
    fast_spatial._MP_METHOD = method
    fast_spatial._MP_KWARGS = classifier_kwargs

    # Infer water year from time axis
    t0_ts = pd.Timestamp(spatial_data['time'][0])
    wy = t0_ts.year + 1 if t0_ts.month >= 10 else t0_ts.year
    if start_date is None:
        start_date = f'{wy}-01-01'
    if end_date is None:
        end_date = f'{wy}-07-01'

    daily_index = pd.date_range(start_date, end_date, freq='D')
    nd = len(daily_index)

    ny_full = spatial_data['swed'].shape[-2]
    nx_full = spatial_data['swed'].shape[-1]
    iy_idx = np.arange(0, ny_full, stride)
    ix_idx = np.arange(0, nx_full, stride)
    ny_out, nx_out = len(iy_idx), len(ix_idx)

    y_out = y_vals[iy_idx] if len(y_vals) == ny_full else y_vals[:ny_out]
    x_out = x_vals[ix_idx] if len(x_vals) == nx_full else x_vals[:nx_out]

    pixels = [(int(iy), int(ix)) for iy in iy_idx for ix in ix_idx]
    n_px = len(pixels)
    print(f'  Classifying {ny_out}×{nx_out} = {n_px:,} pixels '
          f'(stride={stride}, workers={n_workers}) ...')

    t0_classify = _time_mod.time()

    # ---------- classify ----------
    if n_workers > 1:
        # Prevent thread oversubscription in forked children
        _saved_env = {}
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS'):
            _saved_env[var] = os.environ.get(var)
            os.environ[var] = '1'

        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers) as pool:
            raw = list(tqdm(
                pool.imap_unordered(
                    fast_spatial._mp_classify_pixel, pixels, chunksize=64,
                ),
                total=n_px, desc='pixels',
            ))

        # Restore environment
        for var, val in _saved_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
    else:
        raw = [fast_spatial._mp_classify_pixel(px)
               for px in tqdm(pixels, desc='pixels')]

    # Release module-level references
    fast_spatial._MP_DATA = None
    fast_spatial._MP_THRESH = None
    fast_spatial._MP_METHOD = 'original'
    fast_spatial._MP_KWARGS = {}

    classify_secs = _time_mod.time() - t0_classify
    n_classified = sum(1 for r in raw if r is not None)
    print(f'  Classified {n_classified:,} snow pixels in {classify_secs:.1f}s')

    # ---------- assemble ----------
    iy_to_ii = {iy: ii for ii, iy in enumerate(iy_idx)}
    ix_to_jj = {ix: jj for jj, ix in enumerate(ix_idx)}

    phase_code_arr = np.zeros((nd, ny_out, nx_out), dtype=np.int8)
    out_start_arr = np.full((ny_out, nx_out), np.datetime64('NaT'),
                            dtype='datetime64[ns]')
    has_output_arr = np.zeros((ny_out, nx_out), dtype=bool)

    CODE_TO_NAME = {0: 'dry', 1: 'moistening', 2: 'ripening', 3: 'output'}

    for res in raw:
        if res is None:
            continue
        ii = iy_to_ii[res['iy']]
        jj = ix_to_jj[res['ix']]

        # 3h → daily max code
        code_3h = xr.DataArray(
            res['code'], dims=['time'], coords={'time': res['time']},
        )
        code_daily = code_3h.resample(time='1D').max()
        code_daily = code_daily.reindex(time=daily_index.values, fill_value=0)
        phase_code_arr[:, ii, jj] = code_daily.values

        if not np.isnat(res['out_start']):
            has_output_arr[ii, jj] = True
            out_start_arr[ii, jj] = res['out_start']

    # Vectorized name mapping
    phase_name_arr = np.full((nd, ny_out, nx_out), 'dry', dtype='U12')
    for code_val, name in CODE_TO_NAME.items():
        phase_name_arr[phase_code_arr == code_val] = name

    return xr.Dataset({
        'phase_code': xr.DataArray(
            phase_code_arr,
            dims=('time', 'south_north', 'east_west'),
            coords={
                'time': daily_index, 'south_north': iy_idx,
                'east_west': ix_idx,
                'y': ('south_north', y_out), 'x': ('east_west', x_out),
            },
        ),
        'phase_name': xr.DataArray(
            phase_name_arr,
            dims=('time', 'south_north', 'east_west'),
            coords={
                'time': daily_index, 'south_north': iy_idx,
                'east_west': ix_idx,
            },
        ),
        'output_start': xr.DataArray(
            out_start_arr,
            dims=('south_north', 'east_west'),
            coords={
                'south_north': iy_idx, 'east_west': ix_idx,
                'y': ('south_north', y_out), 'x': ('east_west', x_out),
            },
        ),
        'has_output': xr.DataArray(
            has_output_arr,
            dims=('south_north', 'east_west'),
            coords={'south_north': iy_idx, 'east_west': ix_idx},
        ),
    })
