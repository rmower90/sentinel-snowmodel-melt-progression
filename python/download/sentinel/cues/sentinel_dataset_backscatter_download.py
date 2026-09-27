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
#year = int(sys.argv[1]) #input from script
start_date = sys.argv[1]
end_date = sys.argv[2]
resolution = int(sys.argv[3])

#print(f'processing {year}, start date = {start_date}, end date = {end_date}, resolution = {resolution}')
print(f'processing start date = {start_date}, end date = {end_date}, resolution = {resolution}')
#save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/snowmodel_30m/outputs/wo_assim/wy_{year}/s1_backscatter/'
# save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/data/sentinel/cues/{resolution}_m/'
save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/data/sentinel/similar_cues/{resolution}_m/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# # -- create shape.
# lon_min = -119.12975 # 300m
# lon_max = -118.92825 # 300m
# lat_min = 37.54225 # 300m
# lat_max = 37.74375 # 300m

# lon_min = -119.079 # mammoth mountain
# lon_max = -118.979 # mammoth mountain
# lat_min = 37.593 # mammoth mountain
# lat_max = 37.693 # mammoth mountain

loc_lat, loc_lon = 37.29433804403356, -119.19995381496882

lon_min = loc_lon - 0.45# similar cues
lon_max = loc_lon + 0.45 # similar cues
lat_min = loc_lat - 0.05 # similar cues
lat_max = loc_lat + 0.05 # similar cues


lat_point_list = [lat_min, lat_max, lat_max, lat_min, lat_min]
lon_point_list = [lon_min, lon_min, lon_max, lon_max, lon_min]

polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])

shapefile_folder = '../../../../data/shapefiles/'
# geojson = 'cues_shape_mammoth.json'
geojson = 'cues_similar.json'

polygon.to_file(f'{shapefile_folder}{geojson}',driver="GeoJSON")

fn = f'{shapefile_folder}{geojson}'

bbox_gdf = gpd.read_file(fn)
bbox_gdf.attrs = {'filename':fn}


sentinel_ds = create_dataset.get_s1_rtc(bbox_gdf,
                          start_date,
                          end_date,
                          resolution)

if not os.path.exists(save_dir): os.makedirs(save_dir)

sentinel_ds.to_netcdf(f'{save_dir}s1_backscatter_{start_date.replace("-","_")}_{end_date.replace("-","_")}_{resolution}m.nc')

