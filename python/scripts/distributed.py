"""
fig_3.py - Functions for Tuolumne distributed melt phase analysis

This module provides functions for:
- Converting SnowModel curvilinear grids to rectilinear
- Calculating terrain aspect with CORRECTED N/S orientation
- Visualizing melt phases by elevation, aspect, and model

The aspect correction fixes an issue where GDAL-computed aspect had N and S
inverted relative to geographic north in this coordinate system.
"""

import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import pyproj
import os
import xarray as xr
import pandas as pd
import rasterio
from osgeo import gdal


def sm_geog_curvilinear_to_rectilinear(
    ds: xr.Dataset,
    var: str = 'VEG',
    proj_string: str = '+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +type=crs'
) -> xr.DataArray:
    """
    Convert SnowModel curvilinear grid to rectilinear.

    Parameters
    ----------
    ds : xr.Dataset
        SnowModel dataset with XLONG, XLAT coordinates
    var : str
        Variable name to extract
    proj_string : str
        Projection string for output coordinates

    Returns
    -------
    xr.DataArray
        Reprojected data array with rectilinear x, y coordinates
    """
    p = pyproj.Proj(proj_string)
    proj_x, proj_y = p(ds.XLONG.values, ds.XLAT.values, inverse=False)
    lon_2v = proj_x[int(ds.XLAT.shape[0] / 2), :]
    lat_2v = proj_y[:, int(ds.XLAT.shape[1] / 2)]

    da = xr.DataArray(
        data=ds[var].values,
        dims=["y", "x"],
        coords=dict(y=(["y"], lat_2v), x=(["x"], lon_2v))
    )
    da.name = var
    da = da.rio.write_crs(proj_string, inplace=False)
    return da


def calculate_aspect(
    DEM: xr.DataArray,
    aspect_dir: str,
    dem_dir: str,
    dem_type: str,
    aso_start_wy: int,
    aso_end_wy: int
) -> xr.DataArray:
    """
    Calculate aspect from DEM using GDAL.

    Parameters
    ----------
    DEM : xr.DataArray
        Digital elevation model
    aspect_dir : str
        Directory to save aspect outputs
    dem_dir : str
        Directory to save DEM outputs
    dem_type : str
        Type identifier for filenames (e.g., 'SM_100M')
    aso_start_wy : int
        Start water year for filename
    aso_end_wy : int
        End water year for filename

    Returns
    -------
    xr.DataArray
        Aspect values in degrees (0-360, clockwise from north)
    """
    outfile_fname_tif = f'{dem_type}_ASPECT_{aso_start_wy}_{aso_end_wy}.tif'
    outfile_fname_nc = f'{dem_type}_ASPECT_{aso_start_wy}_{aso_end_wy}.nc'
    dem_fname = outfile_fname_tif.replace('ASPECT', 'DEM')

    if not os.path.exists(dem_dir + dem_fname):
        DEM.rio.to_raster(dem_dir + dem_fname)
    if not os.path.exists(aspect_dir):
        os.makedirs(aspect_dir)
    if not os.path.exists(aspect_dir + outfile_fname_tif):
        gdal.DEMProcessing(aspect_dir + outfile_fname_tif, dem_dir + dem_fname, 'aspect')

    with rasterio.open(aspect_dir + outfile_fname_tif) as dataset:
        aspect = dataset.read(1)

    if not os.path.exists(aspect_dir + outfile_fname_nc):
        aspect_da = xr.DataArray(
            data=aspect,
            dims=["y", "x"],
            coords=dict(y=(["y"], DEM.y.values), x=(["x"], DEM.x.values))
        )
        aspect_da.name = 'ASPECT'
        aspect_da = aspect_da.where(aspect_da >= 0)
        aspect_da.to_netcdf(aspect_dir + outfile_fname_nc)
    else:
        aspect_da = xr.open_dataarray(aspect_dir + outfile_fname_nc)

    return aspect_da


def add_terrain_bins_corrected_aspect(
    terrain_ds: xr.Dataset,
    *,
    elev_bin_m: float = 400.0,
    hgt_var: str = "hgt",
    aspect_var: str = "aspect",
) -> xr.Dataset:
    """
    Add terrain bins with CORRECTED N/S aspect swap.

    The original GDAL aspect had N and S inverted relative to geographic north
    for this coordinate system. This function corrects that by swapping N (0)
    and S (2) assignments.

    Parameters
    ----------
    terrain_ds : xr.Dataset
        Dataset containing height and aspect variables
    elev_bin_m : float
        Elevation bin width in meters (default 400m)
    hgt_var : str
        Name of height variable
    aspect_var : str
        Name of aspect variable

    Returns
    -------
    xr.Dataset
        Dataset with added elev_bin, elev_bin_edges, and aspect_bin variables

    Notes
    -----
    Aspect bins:
      0 = N-facing (slope faces north, receives less sun)
      1 = E-facing
      2 = S-facing (slope faces south, receives more sun)
      3 = W-facing
      -1 = invalid
    """
    ds = terrain_ds.load()

    # Elevation bins
    hgt = ds[hgt_var]
    if 'elev' in hgt.dims:
        if 'total' in ds.coords.get('elev', xr.DataArray([])).values:
            hgt2d = hgt.sel(elev='total')
        else:
            hgt2d = hgt.median('elev', skipna=True)
    else:
        hgt2d = hgt

    dz = float(elev_bin_m)
    h = hgt2d.values
    hmin, hmax = np.nanmin(h), np.nanmax(h)
    elev_edges = np.arange(np.floor(hmin / dz) * dz, np.ceil(hmax / dz) * dz + dz, dz)
    elev_bin = (np.digitize(h, elev_edges) - 1).astype(np.int16)

    ds = ds.assign(
        elev_bin=xr.DataArray(elev_bin, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]}),
        elev_bin_edges=xr.DataArray(elev_edges, dims=("elev_bin_edge",)),
    )

    # Aspect bins - CORRECTED for N/S swap
    a = ds[aspect_var].values
    aspect_bin = np.full(a.shape, -1, dtype=np.int16)
    m = np.isfinite(a)
    aa = np.mod(a[m], 360.0)

    # CORRECTED assignments (N and S swapped from original)
    tmp = np.full_like(aa, 2, dtype=np.int16)  # Default to S (was N in original)
    tmp[(aa >= 45) & (aa < 135)] = 1    # E - unchanged
    tmp[(aa >= 135) & (aa < 225)] = 0   # N - SWAPPED (was S=2 in original)
    tmp[(aa >= 225) & (aa < 315)] = 3   # W - unchanged
    # [315,360) + [0,45) stays as S=2 - SWAPPED (was N=0 in original)
    aspect_bin[m] = tmp

    ds = ds.assign(
        aspect_bin=xr.DataArray(aspect_bin, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})
    ).assign_coords(
        aspect_bin_label=xr.DataArray(["N", "E", "S", "W"], dims=("aspect_bin_label_dim",))
    )

    return ds


def verify_aspect_correction(terrain_binned: xr.Dataset) -> None:
    """
    Verify that aspect correction is working by checking elevation gradients.

    Parameters
    ----------
    terrain_binned : xr.Dataset
        Dataset with terrain bins applied
    """
    print("=== Verifying CORRECTED Aspect Bins ===")
    print("\nChecking that elevation gradients match expected directions...")

    hgt = terrain_binned['hgt'].values
    aspect_bin = terrain_binned['aspect_bin'].values

    for ab, label, expected_grad in [(0, 'N', 'higher to S'),
                                      (1, 'E', 'higher to W'),
                                      (2, 'S', 'higher to N'),
                                      (3, 'W', 'higher to E')]:
        mask = aspect_bin == ab
        if mask.sum() < 100:
            continue

        y_idx, x_idx = np.where(mask)
        n_sample = min(100, len(y_idx))
        sample_idx = np.random.choice(len(y_idx), n_sample, replace=False)

        gradients_correct = 0
        for i in sample_idx:
            yi, xi = y_idx[i], x_idx[i]
            if yi < 1 or yi >= hgt.shape[0] - 1 or xi < 1 or xi >= hgt.shape[1] - 1:
                continue
            center = hgt[yi, xi]
            if not np.isfinite(center):
                continue

            north = hgt[yi + 1, xi] if np.isfinite(hgt[yi + 1, xi]) else center
            south = hgt[yi - 1, xi] if np.isfinite(hgt[yi - 1, xi]) else center
            east = hgt[yi, xi + 1] if np.isfinite(hgt[yi, xi + 1]) else center
            west = hgt[yi, xi - 1] if np.isfinite(hgt[yi, xi - 1]) else center

            if ab == 0 and south > center:
                gradients_correct += 1
            elif ab == 1 and west > center:
                gradients_correct += 1
            elif ab == 2 and north > center:
                gradients_correct += 1
            elif ab == 3 and east > center:
                gradients_correct += 1

        pct = gradients_correct / n_sample * 100
        status = "✓" if pct > 60 else "⚠️"
        print(f"  {label}-facing: {pct:.0f}% have expected gradient ({expected_grad}) {status}")


def setup_melt_phase_plot(
    models: list,
    terrain_binned: xr.Dataset,
    figsize_per_model: float = 5.0,
    figsize_per_elev: float = 2.5,
    dpi: int = 150,
) -> dict:
    """
    Set up figure and common variables for melt phase plotting.

    Parameters
    ----------
    models : list
        List of (model_name, dataset) tuples
    terrain_binned : xr.Dataset
        Dataset with terrain bins applied
    figsize_per_model : float
        Figure width per model column
    figsize_per_elev : float
        Figure height per elevation row
    dpi : int
        Figure resolution

    Returns
    -------
    dict
        Dictionary containing fig, outer_gs, and all setup variables
    """
    phase_colors = {0: 'white', 1: 'lightblue', 2: 'palegoldenrod', 3: 'lightcoral', 4: '#c8c8c8'}
    phase_names = {0: 'dry', 1: 'moist', 2: 'ripe', 3: 'output', 4: 'na'}

    elev_bins = list(np.arange(1, terrain_binned.elev_bin_edge.shape[0] - 1))
    aspect_bins = [0, 1, 2, 3]
    aspect_labels = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    elev_edges = terrain_binned['elev_bin_edges'].values

    n_models = len(models)
    n_elev = len(elev_bins)
    n_aspect = len(aspect_bins)

    fig = plt.figure(figsize=(figsize_per_model * n_models, figsize_per_elev * n_elev), dpi=dpi)
    outer_gs = fig.add_gridspec(n_elev, n_models, hspace=0.3, wspace=0.12)

    return {
        'fig': fig,
        'outer_gs': outer_gs,
        'phase_colors': phase_colors,
        'phase_names': phase_names,
        'elev_bins': elev_bins,
        'aspect_bins': aspect_bins,
        'aspect_labels': aspect_labels,
        'elev_edges': elev_edges,
        'n_models': n_models,
        'n_elev': n_elev,
        'n_aspect': n_aspect,
    }


def plot_sm_melt_phases(
    setup: dict,
    models: list,
    terrain_binned: xr.Dataset,
    water_year: int,
    swe_datasets: list = None,
    swe_threshold: float = 0.1,
    sentinel_qc_ds: xr.Dataset = None,
    min_pixels: int = 20,
) -> plt.Figure:
    """
    Plot SnowModel melt phases by elevation, aspect, and model.

    Only has_output=True grid cells are included in the median calculation.

    Parameters
    ----------
    setup : dict
        Setup dictionary from setup_melt_phase_plot()
    models : list
        List of (model_name, dataset) tuples
    terrain_binned : xr.Dataset
        Dataset with terrain bins applied
    water_year : int
        Water year for title
    swe_datasets : list of xr.DataArray, optional
        Daily SWE DataArrays (time, y, x), one per model, aligned with models list.
        When provided, timesteps where bin-median SWE < swe_threshold are shown as 'na'.
    swe_threshold : float
        SWE threshold in meters below which a timestep is marked 'na' (default 0.1)
    sentinel_qc_ds : xr.Dataset, optional
        Sentinel dataset with a boolean `qc_ok` variable (y, x). When provided,
        only pixels where qc_ok=True are included in the aggregation.
    min_pixels : int
        Minimum pixels required to plot (default 20)

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig = setup['fig']
    outer_gs = setup['outer_gs']
    phase_colors = setup['phase_colors']
    phase_names = setup['phase_names']
    elev_bins = setup['elev_bins']
    aspect_bins = setup['aspect_bins']
    aspect_labels = setup['aspect_labels']
    elev_edges = setup['elev_edges']
    n_elev = setup['n_elev']
    n_aspect = setup['n_aspect']

    for row_idx, eb in enumerate(elev_bins):
        eb_lo = int(elev_edges[eb])
        eb_hi = int(elev_edges[eb + 1])

        for col_idx, (model_name, ds) in enumerate(models):
            inner_gs = outer_gs[row_idx, col_idx].subgridspec(n_aspect, 1, hspace=0.08)

            phase_all = np.asarray(ds['phase_code'].values)
            has_output = ds['has_output'].values
            time_vals = np.asarray(ds.time.values)
            finite_any = np.isfinite(phase_all).any(axis=0)

            year = pd.to_datetime(time_vals[0]).year
            xlim_start = np.datetime64(f'{year}-01-01')
            xlim_end = np.datetime64(f'{year}-07-01')

            for asp_idx, ab in enumerate(aspect_bins):
                ax = fig.add_subplot(inner_gs[asp_idx])

                terrain_mask = (
                    (terrain_binned['elev_bin'].values == eb) &
                    (terrain_binned['aspect_bin'].values == ab)
                )

                # Filter: Only include pixels where has_output=True
                sentinel_qc = sentinel_qc_ds['qc_ok'].values if sentinel_qc_ds is not None else True
                qc_mask = terrain_mask & finite_any & has_output & sentinel_qc
                n_valid = int(qc_mask.sum())
                n_total = int((terrain_mask & finite_any).sum())

                if n_valid < min_pixels:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(
                        0.5, 0.5, f'n={n_valid}/{n_total}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6, color='gray'
                    )
                else:
                    phase_numeric = phase_all[:, qc_mask].astype(float)
                    median_phase = np.rint(np.nanmedian(phase_numeric, axis=1))
                    median_phase = np.where(np.isfinite(median_phase), median_phase, 0).astype(int)
                    median_phase = np.clip(median_phase, 0, 3)

                    # Apply SWE mask: timesteps where bin-median SWE < threshold → na (4)
                    if swe_datasets is not None:
                        swe_da = swe_datasets[col_idx]
                        swe_da = swe_da.reindex(time=time_vals, method='nearest', tolerance='1D')
                        swe_vals = swe_da.values  # (time, y, x)
                        bin_swe = swe_vals[:, qc_mask]
                        median_swe = np.nanmedian(bin_swe, axis=1)
                        median_phase = np.where(median_swe < swe_threshold, 4, median_phase)

                    for t in range(len(time_vals) - 1):
                        ax.axvspan(
                            time_vals[t], time_vals[t + 1],
                            facecolor=phase_colors[int(median_phase[t])], edgecolor='none'
                        )
                    ax.axvspan(
                        time_vals[-1], time_vals[-1] + np.timedelta64(1, 'D'),
                        facecolor=phase_colors[int(median_phase[-1])], edgecolor='none'
                    )
                    ax.set_facecolor('white')

                ax.set_xlim(xlim_start, xlim_end)
                ax.set_ylim(0, 1)
                ax.set_yticks([])

                if col_idx == 0:
                    ax.set_ylabel(
                        aspect_labels[ab], fontsize=8, fontweight='bold',
                        rotation=0, labelpad=12, va='center'
                    )

                ax.text(
                    0.99, 0.5, f'n={n_valid}/{n_total}', transform=ax.transAxes,
                    ha='right', va='center', fontsize=5, color='gray', alpha=0.7
                )

                if row_idx == n_elev - 1 and asp_idx == n_aspect - 1:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                    ax.tick_params(axis='x', labelsize=6, rotation=45)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)

            if row_idx == 0:
                first_ax = fig.axes[-n_aspect]
                first_ax.set_title(model_name, fontweight='bold', fontsize=11)

        last_ax = fig.axes[-1]
        last_ax.annotate(
            f'{eb_lo}-{eb_hi} m', xy=(1.04, 2.0), xycoords='axes fraction',
            fontsize=9, fontweight='bold', va='center', rotation=-90
        )

    legend_codes = [0, 1, 2, 3] + ([4] if swe_datasets is not None else [])
    legend_patches = [
        Patch(facecolor=phase_colors[v], edgecolor='gray', label=phase_names[v])
        for v in legend_codes
    ]
    fig.legend(
        handles=legend_patches, loc='lower center', ncol=len(legend_codes), fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        f'WY {water_year} - SnowModel Median Melt Phase\n(Corrected Aspect, has_output=True only)',
        fontweight='bold', fontsize=13, y=1.02
    )

    return fig


def print_summary_stats(
    models: list,
    terrain_binned: xr.Dataset,
    water_year: int,
) -> None:
    """
    Print summary statistics of pixel counts by terrain bin.

    Parameters
    ----------
    models : list
        List of (model_name, dataset) tuples
    terrain_binned : xr.Dataset
        Dataset with terrain bins applied
    water_year : int
        Water year for title
    """
    elev_bins = list(np.arange(1, terrain_binned.elev_bin_edge.shape[0] - 1))
    aspect_bins = [0, 1, 2, 3]
    aspect_labels = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    elev_edges = terrain_binned['elev_bin_edges'].values

    print(f"=== WY {water_year} Summary: Pixel Counts by Terrain Bin ===")
    print(f"\nFiltering to has_output=True pixels only\n")

    for model_name, ds in models:
        print(f"--- {model_name} ---")
        has_output = ds['has_output'].values
        phase_all = np.asarray(ds['phase_code'].values)
        finite_any = np.isfinite(phase_all).any(axis=0)

        for eb in elev_bins:
            eb_lo = int(elev_edges[eb])
            eb_hi = int(elev_edges[eb + 1])

            row_data = []
            for ab in aspect_bins:
                terrain_mask = (
                    (terrain_binned['elev_bin'].values == eb) &
                    (terrain_binned['aspect_bin'].values == ab)
                )
                n_total = (terrain_mask & finite_any).sum()
                n_output = (terrain_mask & finite_any & has_output).sum()
                pct = n_output / n_total * 100 if n_total > 0 else 0
                row_data.append(f"{aspect_labels[ab]}:{n_output}/{n_total}({pct:.0f}%)")

            print(f"  {eb_lo}-{eb_hi}m: {' | '.join(row_data)}")
        print()


_PHASE_NAME_TO_CODE = {'dry': 0, 'moist': 1, 'ripe': 2, 'output': 3, 'na': 4}


def plot_melt_phase_comparison(
    setup: dict,
    columns: list,
    terrain_binned: xr.Dataset,
    water_year: int,
    swe_threshold: float = 0.1,
    sentinel_qc_ds: xr.Dataset = None,
    min_pixels: int = 20,
) -> plt.Figure:
    """
    Plot melt phases for a mix of SnowModel and Sentinel columns side by side.

    Parameters
    ----------
    setup : dict
        Setup dictionary from setup_melt_phase_plot() called with columns as the
        first argument (so n_models = len(columns)).
    columns : list of dict
        Each dict describes one column:
          - 'name'  : str   — column title
          - 'type'  : 'snowmodel' | 'sentinel'
          - 'ds'    : xr.Dataset — phase data
          - 'swe'   : xr.DataArray | None — daily SWE for na masking, None to skip
        SnowModel datasets must have 'phase_code' (int) and 'has_output' (bool).
        Sentinel datasets must have 'phase_name' (str) and 'qc_ok' (bool).
    terrain_binned : xr.Dataset
        Dataset with terrain bins applied.
    water_year : int
        Water year for title.
    swe_threshold : float
        Bin-median SWE (m) below which a timestep is marked 'na'. Ignored if
        column 'swe' is None.
    sentinel_qc_ds : xr.Dataset, optional
        Sentinel dataset with 'qc_ok' (y, x). When provided, SnowModel pixel
        aggregation is additionally filtered to qc_ok=True pixels.
    min_pixels : int
        Minimum pixels required to plot a bin (default 20).

    Returns
    -------
    plt.Figure
    """
    fig = setup['fig']
    outer_gs = setup['outer_gs']
    phase_colors = setup['phase_colors']
    phase_names = setup['phase_names']
    elev_bins = setup['elev_bins']
    aspect_bins = setup['aspect_bins']
    aspect_labels = setup['aspect_labels']
    elev_edges = setup['elev_edges']
    n_elev = setup['n_elev']
    n_aspect = setup['n_aspect']

    sentinel_qc = sentinel_qc_ds['qc_ok'].values if sentinel_qc_ds is not None else None

    for row_idx, eb in enumerate(elev_bins):
        eb_lo = int(elev_edges[eb])
        eb_hi = int(elev_edges[eb + 1])

        for col_idx, col in enumerate(columns):
            col_name = col['name']
            col_type = col['type']
            ds = col['ds']
            swe_da = col.get('swe', None)

            inner_gs = outer_gs[row_idx, col_idx].subgridspec(n_aspect, 1, hspace=0.08)

            # Build phase_code array (time, n_pixels_flat)
            if col_type == 'snowmodel':
                phase_all = np.asarray(ds['phase_code'].values)        # (time, y, x)
                has_out = ds['has_output'].values                        # (y, x)
                finite_any = np.isfinite(phase_all.astype(float)).any(axis=0)
                col_qc = has_out & finite_any
                if sentinel_qc is not None:
                    col_qc = col_qc & sentinel_qc
            else:  # sentinel
                phase_name_arr = ds['phase_name'].values                # (time, y, x) str
                phase_all = np.vectorize(_PHASE_NAME_TO_CODE.get)(
                    phase_name_arr, 0
                ).astype(np.int8)
                col_qc = ds['qc_ok'].values                             # (y, x)

            time_vals = np.asarray(ds.time.values)
            year = pd.to_datetime(time_vals[0]).year
            xlim_start = np.datetime64(f'{year}-01-01')
            xlim_end = np.datetime64(f'{year}-07-01')

            # Align SWE to this column's time axis if provided
            if swe_da is not None:
                swe_aligned = swe_da.reindex(
                    time=time_vals, method='nearest', tolerance='1D'
                ).values  # (time, y, x)
            else:
                swe_aligned = None

            for asp_idx, ab in enumerate(aspect_bins):
                ax = fig.add_subplot(inner_gs[asp_idx])

                terrain_mask = (
                    (terrain_binned['elev_bin'].values == eb) &
                    (terrain_binned['aspect_bin'].values == ab)
                )
                qc_mask = terrain_mask & col_qc
                n_valid = int(qc_mask.sum())
                n_total = int(terrain_mask.sum())

                if n_valid < min_pixels:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(
                        0.5, 0.5, f'n={n_valid}/{n_total}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6, color='gray'
                    )
                else:
                    phase_numeric = phase_all[:, qc_mask].astype(float)
                    median_phase = np.rint(np.nanmedian(phase_numeric, axis=1))
                    median_phase = np.where(
                        np.isfinite(median_phase), median_phase, 0
                    ).astype(int)
                    median_phase = np.clip(median_phase, 0, 4)

                    # Bin-aggregate SWE mask → na (4)
                    if swe_aligned is not None:
                        bin_swe = swe_aligned[:, qc_mask]
                        median_swe = np.nanmedian(bin_swe, axis=1)
                        median_phase = np.where(
                            median_swe < swe_threshold, 4, median_phase
                        )

                    for t in range(len(time_vals) - 1):
                        ax.axvspan(
                            time_vals[t], time_vals[t + 1],
                            facecolor=phase_colors[int(median_phase[t])],
                            edgecolor='none',
                        )
                    ax.axvspan(
                        time_vals[-1],
                        time_vals[-1] + np.timedelta64(1, 'D'),
                        facecolor=phase_colors[int(median_phase[-1])],
                        edgecolor='none',
                    )
                    ax.set_facecolor('white')

                ax.set_xlim(xlim_start, xlim_end)
                ax.set_ylim(0, 1)
                ax.set_yticks([])

                if col_idx == 0:
                    ax.set_ylabel(
                        aspect_labels[ab], fontsize=8, fontweight='bold',
                        rotation=0, labelpad=12, va='center'
                    )

                ax.text(
                    0.99, 0.5, f'n={n_valid}/{n_total}', transform=ax.transAxes,
                    ha='right', va='center', fontsize=5, color='gray', alpha=0.7
                )

                if row_idx == n_elev - 1 and asp_idx == n_aspect - 1:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                    ax.tick_params(axis='x', labelsize=6, rotation=45)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)

            if row_idx == 0:
                first_ax = fig.axes[-n_aspect]
                first_ax.set_title(col_name, fontweight='bold', fontsize=11)

        last_ax = fig.axes[-1]
        last_ax.annotate(
            f'{eb_lo}-{eb_hi} m', xy=(1.04, 2.0), xycoords='axes fraction',
            fontsize=9, fontweight='bold', va='center', rotation=-90
        )

    has_swe_mask = any(col.get('swe') is not None for col in columns)
    legend_codes = [0, 1, 2, 3] + ([4] if has_swe_mask else [])
    legend_patches = [
        Patch(facecolor=phase_colors[v], edgecolor='gray', label=phase_names[v])
        for v in legend_codes
    ]
    fig.legend(
        handles=legend_patches, loc='lower center', ncol=len(legend_codes),
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    sentinel_qc_note = f', SM filtered to Sentinel qc_ok' if sentinel_qc_ds is not None else ''
    fig.suptitle(
        f'WY {water_year} - Median Melt Phase by Elevation × Aspect{sentinel_qc_note}',
        fontweight='bold', fontsize=13, y=1.02
    )

    return fig


def load_swe_t00(
    sm_dir: str,
    water_year: int,
    ref_da: xr.DataArray,
    clip_gdf,
    start_date: str = None,
    end_date: str = None,
) -> xr.DataArray:
    """
    Load daily T00 SWE snapshots from SnowModel 3-hourly output files.

    Selects midnight (T00) files only, assigns projected coordinates from a
    reference DataArray, and clips to the watershed boundary.

    Parameters
    ----------
    sm_dir : str
        Root directory for the model variant (e.g. ml_biased/)
    water_year : int
        Water year (e.g. 2020)
    ref_da : xr.DataArray
        Reference DataArray with projected y/x coordinates (e.g. elev_da).
        Used to assign spatial coordinates to the SWE grid.
    clip_gdf : GeoDataFrame
        Watershed boundary for clipping
    start_date : str, optional
        ISO date string (YYYY-MM-DD). Defaults to '{water_year-1}-10-01'.
    end_date : str, optional
        ISO date string (YYYY-MM-DD). Defaults to '{water_year}-09-01'.

    Returns
    -------
    xr.DataArray
        Daily SWE (meters) with dims (time, y, x), clipped to watershed.
    """
    start_date = start_date or f'{water_year - 1}-10-01'
    end_date = end_date or f'{water_year}-09-01'
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    pattern = f'{sm_dir}wy_{water_year}/netcdf/sm_swed_*000000.nc'
    all_files = sorted(glob.glob(pattern))

    # Filter to date range by parsing timestamp from filename
    def _file_date(fpath):
        stem = os.path.basename(fpath).replace('sm_swed_', '').replace('.nc', '')
        return pd.Timestamp(stem[:8])

    files = [f for f in all_files if start_ts <= _file_date(f) <= end_ts]

    if not files:
        raise FileNotFoundError(
            f'No T00 SWE files found in {pattern} between {start_date} and {end_date}'
        )

    swe_ds = xr.open_mfdataset(files, combine='by_coords')

    # Assign projected coordinates from reference DataArray
    swe_da = xr.DataArray(
        swe_ds['swed'].values,
        dims=['time', 'y', 'x'],
        coords={
            'time': swe_ds.time.values,
            'y': ref_da.y.values,
            'x': ref_da.x.values,
        }
    )
    swe_da = swe_da.rio.write_crs(ref_da.rio.crs)
    swe_da = swe_da.rio.clip(clip_gdf.geometry)

    return swe_da
