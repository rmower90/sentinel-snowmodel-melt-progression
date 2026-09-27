# sentinel_spatial_melt.py
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import importlib
import sys
import rioxarray as rxr
from pathlib import Path
import xarray as xr
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# sys.path.insert(1, '../scripts/')

import directories as directories
import metadata as metadata
import load_data as load_data
import melt_phases as melt_phases


"""
    SYS INPUTS
"""
domain = sys.argv[1]

"""
    CONFIG
"""
# ── Data root ────────────────────────────────────────────────────────────────

if domain == "tuolumne":
    domain_cfg = metadata.load_yaml(
        Path("../../configs/Tuolumne.yaml")
    )
else:
    domain_cfg = metadata.load_yaml(
        Path("../../configs/CUES.yaml")
    )

# This already resolves through 100_m/
SENTINEL_ROOT = metadata.get_sentinel_fpath(
    domain_cfg,
    "100m",
)

OUTPUT_DIR = domain_cfg["sentinel_fpath"]["melt_phases"]

os.makedirs(f"{OUTPUT_DIR}vv/", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}rc/", exist_ok=True)


def bscatter_path(year: int) -> str:
    return f"{SENTINEL_ROOT}{year}/bscatter/"


def reference_path(year: int) -> str:
    return f"{SENTINEL_ROOT}{year}/melt_thresh/reference/"


def zarr_path(year: int) -> str:
    return f"{SENTINEL_ROOT}{year}/melt_thresh/data/datasets/"

YEARS = [2017, 2018, 2019, 2020]

# ── Method parameters (match fig_2) ──────────────────────────────────────────
WET_THRESH_DB = -2.0   # Rc threshold for wet snow (dB)
K       = 0.5          # weight between VH and VV in Rc
THETA1  = 20           # lower incidence angle limit (deg)
THETA2  = 45           # upper incidence angle limit (deg)

ORBITS = (64, 137, 144)   # all available orbits

"""
    LOAD SHAPE
"""

geog_fpath,aso_crs = metadata.get_shapefile_fpath(domain_cfg,"shapefile_fpath","zoom_1")

# crs.
aso_crs = domain_cfg["crs"]["epsg"]

geog_gdf = gpd.read_file(geog_fpath)
proj_gdf = geog_gdf.to_crs(f"EPSG:{aso_crs}")

print(f"geog_gdf CRS : EPSG:{aso_crs}")
print(f"geog_gdf bounds: {proj_gdf.total_bounds}")

"""
    LOAD BACKSCATTER / REFERENCE
"""
bscatter = {}
for yr in YEARS:
    print(f"Loading backscatter {yr} ...", end=" ")
    bscatter[yr] = load_data.load_sentinel_backscatter(
        bscatter_path(yr),
        proj_gdf,
        wy=None,
    )
    print(f"dims={dict(bscatter[yr].dims)}")

reference = {}
for yr in YEARS:
    # reference for year Y uses the previous year's summer as dry-snow baseline
    print(f"Loading reference {yr} (from {yr-1}) ...", end=" ")
    reference[yr] = load_data.load_sentinel_reference(
        reference_path(yr),
        aso_crs,
        proj_gdf,
    )
    print(f"dims={dict(reference[yr].dims)}")

"""
    METHOD CONFIGURATIONS
"""
# VV method config (Marin minima + QC)
cfg = melt_phases.MarinS1VVCfg()

# Weighted method config
cfg_w = melt_phases.WeightedMinimaCfg(
    k=K,
    theta1=THETA1,
    theta2=THETA2,
    wet_thresh_db=WET_THRESH_DB,
    cfg_marin=melt_phases.MarinS1VVCfg(
        min_stability_days=28,
        max_group_spread_days=28,
    ),
)

print("VV cfg     :", cfg)
print("Weighted cfg:", cfg_w)


"""
    RUN SPATIAL
"""

for yr in [2017, 2018, 2019, 2020]:
    print(f"\n{'='*60}")
    print(f"  Processing {yr}")
    print(f"{'='*60}")

    # VV method
    vv_phase = melt_phases.run_vv_spatial(
        bscatter[yr],
        reference[yr],
        year=yr,
        cfg=cfg,
        start_date=f"{yr}-01-01",
        end_date=f"{yr}-07-01",
    )

    # Weighted method
    rc_spatial = melt_phases.build_spatial_rc_from_files(
        zarr_path(yr),
        clip_gdf=proj_gdf,
        orbits=ORBITS,
        k=K, theta1=THETA1, theta2=THETA2, wet_thresh_db=WET_THRESH_DB,
    )

    wt_phase = melt_phases.run_weighted_spatial(
        rc_spatial,
        year=yr,
        cfg_w=cfg_w,
        start_date=f"{yr}-01-01",
        end_date=f"{yr}-07-01",
    )

    print(f"  VV  output_start QC-ok pixels: {vv_phase['qc_ok'].values.sum()}")
    print(f"  Wt  output_start QC-ok pixels: {wt_phase['qc_ok'].values.sum()}")

    vv_phase.to_netcdf(f'{OUTPUT_DIR}/vv/{yr}_wetness_phase.nc')
    wt_phase.to_netcdf(f'{OUTPUT_DIR}/rc/{yr}_wetness_phase.nc')

print("\nAll years complete.")