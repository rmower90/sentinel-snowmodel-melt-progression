# directories.py
"""
    Script sets up directory filepaths.
"""
import xarray as xr
import os 
import rioxarray as rxr
from pyproj import CRS, Transformer
import sys
import geopandas as gpd
import numpy as np
import warnings
import rasterio
warnings.filterwarnings("ignore")


def load_campaign_directories(aso_site_name: str,
                              isProcessed: bool = False,
                  ):
    """
    Loads relavent file directories
    Input:
      aso_site_name - python string for basin abbreviation.
    Output:
      aso_dir - python string for absolute filepath of aso directory.
      shape_dir - python string for absolute filepath of shape directory.
      dem_dir - python string for absolute filepath of dem directory.
      states_fpath - python string for absolute filepath of states shapefile.
    """
    campaign_data_dir = '/glade/campaign/ral/hap/rmower/aso/data/'
    if isProcessed:
        aso_dir = f'{campaign_data_dir}aso/{aso_site_name}/processed/'
        shape_dir = f'{campaign_data_dir}shape/{aso_site_name}/processed/'
        dem_dir = f'{campaign_data_dir}dem/{aso_site_name}/processed/'
        aspect_dir = f'{campaign_data_dir}aspect/{aso_site_name}/processed/'
        slope_dir = f'{campaign_data_dir}slope/{aso_site_name}/processed/'
    else:
        aso_dir = f'{campaign_data_dir}aso/{aso_site_name}/raw/'
        shape_dir = f'{campaign_data_dir}shape/{aso_site_name}/raw/'
        dem_dir = f'{campaign_data_dir}dem/{aso_site_name}/raw/'
        aspect_dir = f'{campaign_data_dir}aspect/{aso_site_name}/raw/'
        slope_dir = f'{campaign_data_dir}slope/{aso_site_name}/raw/'

    states_fpath = f'{campaign_data_dir}shape/states/cb_2018_us_state_20m.shp'
    
    return aso_dir,shape_dir,dem_dir,states_fpath,aspect_dir,slope_dir



if __name__ =="__main__":
    if len(sys.argv) < 2:
        print("Usage: python directories.py <aso_site_name>")
        sys.exit(1)
    # get inputs.
    aso_site_name = sys.argv[1]
    # load geography.
    shape_geog_gdf = load_campaign_directories(aso_site_name)






