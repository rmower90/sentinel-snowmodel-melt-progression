#!/usr/bin/env python
"""
sm_spatial_melt.py — Spatial melt-phase classification for SnowModel grids.

Loads SnowModel multilayer outputs for a given model variant and water year,
classifies melt phases spatially using fast_spatial, and writes the result
to a NetCDF file.

Supports two classification methods:
  - 'original'  : classify_snowmodel_melt_phases_3h_full()
  - 'revamp'    : classify_snowmodel_melt_phases_3h_full_revamp()

Usage
-----
    python sm_spatial_melt.py --model corrected --wy 2017 --method revamp
    python sm_spatial_melt.py --model biased --wy 2017 --method original --stride 4
    python sm_spatial_melt.py --help
"""

import argparse
import os
import sys
import time as _time_mod
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

# Ensure scripts/ is on the path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(1, '../scripts/')
import fast_spatial
import melt_phases  # noqa: F401 — needed by fast_spatial workers
import metadata

# ── Paths ────────────────────────────────────────────────────────────────────
tuolumne_cfg = metadata.load_yaml(Path("../../configs/Tuolumne.yaml"))
SM_BIASED_ROOT = metadata.get_snowmodel_fpath(tuolumne_cfg, "biased")
SM_CORRECTED_ROOT = metadata.get_snowmodel_fpath(tuolumne_cfg, "corrected")

VARIANTS = {
    'biased': SM_BIASED_ROOT,
    'corrected': SM_CORRECTED_ROOT,
}

SNOWMODEL_DICT = {2017: 'wy_2017', 2018: 'wy_2018',
                  2019: 'wy_2019', 2020: 'wy_2020'}

SM_PROJ = '+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +type=crs'


# ── Helper ───────────────────────────────────────────────────────────────────
def sm_get_projected_axes(sm_output_dir, wy=2017, proj_string=SM_PROJ):
    """Return (y_vals, x_vals) projected coordinate arrays for the SM domain."""
    wy_dir = SNOWMODEL_DICT[wy]
    geog_path = f'{sm_output_dir}{wy_dir}/geography/ctrl_geog.nc'
    ctrl_ds = xr.load_dataset(geog_path)

    p = pyproj.Proj(proj_string)
    proj_x, proj_y = p(ctrl_ds.XLONG.values, ctrl_ds.XLAT.values,
                       inverse=False)

    y_vals = proj_y[:, int(ctrl_ds.XLAT.shape[1] // 2)]
    x_vals = proj_x[int(ctrl_ds.XLAT.shape[0] // 2), :]
    return y_vals, x_vals


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Spatial melt-phase classification for SnowModel grids.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', choices=list(VARIANTS),
                        default='corrected',
                        help='Model variant.')
    parser.add_argument('--wy', type=int, choices=list(SNOWMODEL_DICT),
                        default=2017,
                        help='Water year.')
    parser.add_argument('--method', choices=['original', 'revamp'],
                        default='revamp',
                        help='Classification method.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Spatial stride (1 = full resolution).')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers.')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Daily output start date (YYYY-MM-DD). '
                             'Defaults to {wy}-01-01.')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Daily output end date (YYYY-MM-DD). '
                             'Defaults to {wy}-09-01.')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for NetCDF files. '
                             'Defaults to {model_root}/melt_phases/.')

    # Classifier parameters (revamp)
    parser.add_argument('--water-thresh', type=float, default=0.0035,
                        help='Water threshold (original method).')
    parser.add_argument('--water-thresh-surface', type=float, default=0.0035,
                        help='Surface liquid threshold (revamp).')
    parser.add_argument('--water-thresh-internal', type=float, default=0.0001,
                        help='Internal liquid threshold (revamp).')
    parser.add_argument('--min-runoff-rate', type=float, default=0.001,
                        help='Min runoff rate for output detection (revamp).')
    parser.add_argument('--output-merge-dry-hours', type=float, default=48.0,
                        help='Merge output gaps shorter than this (revamp).')
    parser.add_argument('--ripen-gap-days', type=float, default=7.0,
                        help='Max gap for between-output ripening (revamp).')

    args = parser.parse_args()

    sm_dir = VARIANTS[args.model]
    start_date = args.start_date or f'{args.wy}-01-01'
    end_date = args.end_date or f'{args.wy}-09-01'
    outdir = args.outdir or f'{sm_dir}melt_phases/'

    print(f'\n{"="*60}')
    print(f'  sm_spatial_melt.py')
    print(f'  Model    : {args.model}')
    print(f'  WY       : {args.wy}')
    print(f'  Method   : {args.method}')
    print(f'  Stride   : {args.stride}')
    print(f'  Workers  : {args.workers}')
    print(f'  Dates    : {start_date} to {end_date}')
    print(f'{"="*60}\n')

    # ── 1. Get projected coordinate axes ──
    y_vals, x_vals = sm_get_projected_axes(sm_dir, wy=args.wy)
    print(f'SM grid: {len(y_vals)} × {len(x_vals)}')

    # ── 2. Load spatial data to memory ──
    print(f'\nLoading {args.model} WY{args.wy} to memory ...')
    t0 = _time_mod.time()
    spatial_data = fast_spatial.load_wy_to_memory(sm_dir, args.wy,
                                                  SNOWMODEL_DICT)
    print(f'Load time: {_time_mod.time() - t0:.1f}s\n')

    # ── 3. Build classifier kwargs ──
    if args.method == 'revamp':
        classifier_kwargs = dict(
            water_thresh_surface=args.water_thresh_surface,
            water_thresh_internal=args.water_thresh_internal,
            min_runoff_rate=args.min_runoff_rate,
            output_merge_dry_hours=args.output_merge_dry_hours,
            ripen_gap_days=args.ripen_gap_days,
        )
        print(f'Revamp params: {classifier_kwargs}')
    else:
        classifier_kwargs = {}
        print(f'Original params: water_thresh={args.water_thresh}')

    # ── 4. Classify spatial ──
    print(f'\nClassifying ...')
    t0 = _time_mod.time()
    sm_spatial = fast_spatial.classify_spatial_parallel(
        spatial_data,
        y_vals=y_vals,
        x_vals=x_vals,
        water_thresh=args.water_thresh,
        start_date=start_date,
        end_date=end_date,
        stride=args.stride,
        n_workers=args.workers,
        method=args.method,
        **classifier_kwargs,
    )
    classify_secs = _time_mod.time() - t0
    print(f'Classification time: {classify_secs:.1f}s')

    n_out = int(sm_spatial['has_output'].values.sum())
    n_tot = sm_spatial['has_output'].size
    print(f'Output pixels: {n_out:,} / {n_tot:,} ({100*n_out/n_tot:.1f}%)')

    # ── 5. Write to disk ──
    os.makedirs(outdir, exist_ok=True)
    wt_str = str(args.water_thresh_surface).replace('.', '_')
    stride_str = f'_stride{args.stride}' if args.stride > 1 else ''
    fname = f'sm_melt_{args.wy}_wt_{wt_str}{stride_str}.nc'
    out_path = os.path.join(outdir, fname)

    # Add metadata attributes
    sm_spatial.attrs['model'] = args.model
    sm_spatial.attrs['water_year'] = args.wy
    sm_spatial.attrs['method'] = args.method
    sm_spatial.attrs['stride'] = args.stride
    sm_spatial.attrs['water_thresh'] = args.water_thresh
    if args.method == 'revamp':
        for k, v in classifier_kwargs.items():
            sm_spatial.attrs[k] = v

    # Drop string variable for NetCDF compatibility (phase_code is sufficient)
    ds_out = sm_spatial.drop_vars('phase_name', errors='ignore')
    ds_out.to_netcdf(out_path)
    print(f'\nWritten: {out_path}')
    print(f'Size: {os.path.getsize(out_path) / 1e6:.1f} MB')
    print('\nDone.')


if __name__ == '__main__':
    main()
