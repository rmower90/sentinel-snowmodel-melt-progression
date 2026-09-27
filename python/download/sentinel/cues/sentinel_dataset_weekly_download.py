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
resolution = int(sys.argv[1])
year = int(sys.argv[2]) #input from script
domain = sys.argv[2]

print(f'domain {domain}; processing {year}; resolution {resolution}')


if domain == 'cues':
    pass
elif domain == 'tuolumne':
    pass
else:
    print('Passed incorrect domain! Please resubmit!')
    sys.exit()
save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/data/sentinel/{domain}/{resolution}_m/{year}/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
reference_scenes_aggregation_technique = 'median' # mean / median / max
time_reference_scenes = slice(f'{year}-07-15',f'{year}-09-01') 

## -- create shape.
lon_min = -119.95
lon_max = -119.20

lat_min = 37.71
lat_max = 38.25

lat_point_list = [lat_min, lat_max, lat_max, lat_min, lat_min]
lon_point_list = [lon_min, lon_min, lon_max, lon_max, lon_min]

polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])

shapefile_folder = 'data/shapefiles/'
geojson = 'tuolumne_shape.json'

polygon.to_file(f'{shapefile_folder}{geojson}',driver="GeoJSON")

fn = f'{shapefile_folder}{geojson}'

bbox_gdf = gpd.read_file(fn)
bbox_gdf.attrs = {'filename':fn}


## download data
weekly_dates = np.arange(f'{year}-05-28', f'{year}-07-01', 7, dtype='datetime64[D]')
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
