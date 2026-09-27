import geopandas as gpd
import create_dataset
import numpy as np
from shapely.geometry import Polygon
import xarray as xr
import zarr
import os

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/glade/u/home/rossamower/work/scripts/snowmodel')

import intersect
import vars_3D

## -- inputs.
year = int(sys.argv[1]) #input from script
resolution = int(sys.argv[2]) #input from script
print(f'processing wy: {year}; resolution {resolution}')

save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/data/sentinel/tuolumne/{year}/'
reference_scenes_aggregation_technique = 'median' # mean / median / max
time_reference_scenes = slice(f'{year}-08-01',f'{year}-09-01') 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

shapefile_folder = '../../../../data/shapefiles/'
geojson = 'tuol_zoom_geog.json'

fn = f'{shapefile_folder}{geojson}'

bbox_gdf = gpd.read_file(fn)
bbox_gdf.attrs = {'filename':fn}


## download data
weekly_dates = np.arange(f'{year-1}-10-01', f'{year}-07-15', 7, dtype='datetime64[D]')
for date in weekly_dates:
    time_target_scene = str(date)
    print(time_target_scene)
    ## download data ##
    dataset = create_dataset.create_dataset(bbox_gdf, 
                                        year, 
                                        time_reference_scenes, 
                                        time_target_scene, 
                                        reference_scenes_aggregation_technique, 
                                        resolution,
                                        save_dir)
    ## output
    if not os.path.exists(f'{save_dir}data/'):
        os.mkdir(f'{save_dir}data/')
    if not os.path.exists(f'{save_dir}data/datasets'):
        os.mkdir(f'{save_dir}data/datasets')
    dataset.to_zarr(f'{save_dir}data/datasets/{geojson.split(".")[0]}_{year}_tar_{time_target_scene}_ref_{time_reference_scenes.start}_{time_reference_scenes.stop}_{reference_scenes_aggregation_technique}_{resolution}.zarr',mode='w')

    dataframe = create_dataset.dataset_to_dataframe(dataset) 
    
    if not os.path.exists(f'{save_dir}data/dataframes'):
        os.mkdir(f'{save_dir}data/dataframes')
    dataframe.to_parquet(f'{save_dir}data/dataframes/{geojson.split(".")[0]}_{year}_tar_{time_target_scene}_ref_{time_reference_scenes.start}_{time_reference_scenes.stop}_{reference_scenes_aggregation_technique}_{resolution}.parquet')
    print(f'finished {time_target_scene}')
    print('')
