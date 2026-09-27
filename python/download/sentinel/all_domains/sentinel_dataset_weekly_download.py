import geopandas as gpd
import create_dataset
import numpy as np
from shapely.geometry import Polygon
import xarray as xr
import zarr
import os
from pathlib import Path

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../../../scripts/')

import directories as directories
import metadata as metadata
import load_data as load_data
import plotting as plotting

## -- inputs.

start_date = sys.argv[1]
end_date = sys.argv[2]
resolution = int(sys.argv[3])
domain = sys.argv[4]
refprevyr = bool(int(sys.argv[5])) # if True; select July 15-Sept1 of previous year. If False; select July 15-Sept1 of current year.

year = int(end_date[0:4])

print(f'domain {domain}; processing {year}; resolution {resolution}; previous year reference period {refprevyr}')

## load metadata
# load config
## --------------- CUES
cues_cfg = metadata.load_yaml(Path(f"../../../../configs/CUES.yaml"))
# load point geometry
cues_lat,cues_lon,cues_x,cues_y,cues_crs = metadata.get_point_geography(cues_cfg,'cues')
sesame_lat,sesame_lon,sesame_x,sesame_y,sesame_crs = metadata.get_point_geography(cues_cfg,'sesame')
mhp_lat,mhp_lon,mhp_x,mhp_y,mhp_crs = metadata.get_point_geography(cues_cfg,'mhp')
# load snowmodel filepath
sm_cues_biased_fpath = metadata.get_snowmodel_fpath(cues_cfg,'biased')
sm_cues_corrected_fpath = metadata.get_snowmodel_fpath(cues_cfg,'corrected')
# load snowmodel filepath
sm_cues_OBS_swe_fpath = metadata.get_cuesOBS_fpath(cues_cfg,'swe')
sm_cues_OBS_sweMelt_fpath = metadata.get_cuesOBS_fpath(cues_cfg,'swe_melt')

## --------------- Tuolumne
tuolumne_cfg = metadata.load_yaml(Path(f"../../../../configs/Tuolumne.yaml"))
tuolumneGauge_lat,tuolumneGauge_lon,tuolumneGauge_x,tuolumneGauge_y,tuolumneGauge_crs = metadata.get_point_geography(tuolumne_cfg,'tuolumne_gauge')
lyellGauge_lat,lyellGauge_lon,lyellGauge_x,lyellGauge_y,lyellGauge_crs = metadata.get_point_geography(tuolumne_cfg,'lyell_gauge')
danaGauge_lat,danaGauge_lon,danaGauge_x,danaGauge_y,danaGauge_crs = metadata.get_point_geography(tuolumne_cfg,'dana_gauge')
tuolumnePillow_lat,tuolumnePillow_lon,tuolumnePillow_x,tuolumnePillow_y,tuolumnePillow_crs = metadata.get_point_geography(tuolumne_cfg,'tuolumne_pillow')
danaPillow_lat,danaPillow_lon,danaPillow_x,danaPillow_y,danaPillow_crs = metadata.get_point_geography(tuolumne_cfg,'dana_pillow')

# cues
mammoth_geog_fpath,aso_crs = metadata.get_shapefile_fpath(cues_cfg,"shapefile_fpath","zoom_1")
cues_geog_fpath,aso_crs = metadata.get_shapefile_fpath(cues_cfg,"shapefile_fpath","zoom_2")
# tuolumne
tuolumne_geog_fpath,aso_crs = metadata.get_shapefile_fpath(tuolumne_cfg,"shapefile_fpath","zoom_1")
dana_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'dana_watershed')
lyell_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'lyell_watershed')
tuolumne_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'tuolumne_watershed')
# read shapefile
## mammoth
mammoth_geog_gdf = gpd.read_file(mammoth_geog_fpath)
mammoth_proj_gdf = mammoth_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## cues
cues_geog_gdf = gpd.read_file(cues_geog_fpath)
cues_proj_gdf = cues_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## tuolumne
tuolumne_geog_gdf = gpd.read_file(tuolumne_geog_fpath)
tuolumne_proj_gdf = tuolumne_geog_gdf.to_crs(f'EPSG:{aso_crs}')


print('mammoth_geog_fpath',mammoth_geog_fpath)


if domain == 'cues':
    bbox_gdf = mammoth_geog_gdf
    bbox_gdf.attrs = {'filename':mammoth_geog_fpath}
    geojson = 'mammoth.json'
elif domain == 'tuolumne':
    bbox_gdf = tuolumne_geog_gdf
    bbox_gdf.attrs = {'filename':tuolumne_geog_fpath}
    geojson = 'tuolumne.json'
elif domain == 'grand_mesa':
    bbox_gdf = grand_mesa_geog_gdf
    bbox_gdf.attrs = {'filename':grand_mesa_geog_fpath}
    geojson = 'grand_mesa.json'
else:
    print('Passed incorrect domain! Please resubmit!')
    sys.exit()
save_dir = f'/glade/u/home/rossamower/rmower/pixel/sentinel/{domain}/{resolution}_m/{year}/'
if refprevyr: # previous year reference period
    save_dir_meltThresh = f'{save_dir}melt_thresh/'
else:
    save_dir_meltThresh = f'{save_dir}melt_thresh_following_ref/'

save_dir_bscatter = f'{save_dir}bscatter/'
reference_dir = f'{save_dir_meltThresh}reference/'

if not os.path.exists(save_dir): os.makedirs(save_dir)


reference_scenes_aggregation_technique = 'median' # mean / median / max

if refprevyr:
    time_reference_scenes = slice(f'{year-1}-07-15',f'{year-1}-09-01') 
else:
    time_reference_scenes = slice(f'{year}-07-15',f'{year}-09-01')


## download data

## straight backscatter
sentinel_ds = create_dataset.get_s1_rtc(bbox_gdf,
                          start_date,
                          end_date,
                          resolution)


if not os.path.exists(save_dir_bscatter): os.makedirs(save_dir_bscatter)
sentinel_ds.to_netcdf(f'{save_dir_bscatter}s1_backscatter_{start_date.replace("-","_")}_{end_date.replace("-","_")}_{resolution}m.nc')

## erics method
weekly_dates = np.arange(f'{year-1}-10-01', f'{year}-10-01', 7, dtype='datetime64[D]')
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
                                        save_dir_meltThresh)
    ## output
    if not os.path.exists(f'{save_dir_meltThresh}data/'):
        os.mkdir(f'{save_dir_meltThresh}data/')
    if not os.path.exists(f'{save_dir_meltThresh}data/datasets'):
        os.mkdir(f'{save_dir_meltThresh}data/datasets')
    dataset.to_zarr(f'{save_dir_meltThresh}data/datasets/{geojson.split(".")[0]}_{year}_tar_{time_target_scene}_ref_{time_reference_scenes.start}_{time_reference_scenes.stop}_{reference_scenes_aggregation_technique}_{resolution}.zarr',mode='w')

    dataframe = create_dataset.dataset_to_dataframe(dataset) 
    
    if not os.path.exists(f'{save_dir_meltThresh}data/dataframes'):
        os.mkdir(f'{save_dir_meltThresh}data/dataframes')
    dataframe.to_parquet(f'{save_dir_meltThresh}data/dataframes/{geojson.split(".")[0]}_{year}_tar_{time_target_scene}_ref_{time_reference_scenes.start}_{time_reference_scenes.stop}_{reference_scenes_aggregation_technique}_{resolution}.parquet')
    print(f'finished {time_target_scene}')
    print('')

## reference image.
if not os.path.exists(reference_dir): os.makedirs(reference_dir)
weekly_dates = np.arange(f'{year-1}-10-01', f'{year}-10-01', 7, dtype='datetime64[D]')
for date in weekly_dates:
    time_target_scene = str(date)

    composite_reference_scene,ratio_images = create_dataset.get_composite_images(bbox_gdf, year, time_reference_scenes, time_target_scene, reference_scenes_aggregation_technique, resolution)
    reference_ds = composite_reference_scene.to_dataset(name='reference_bscatter')
    reference_ds = reference_ds.rio.write_crs(bbox_gdf.estimate_utm_crs().to_epsg())
    reference_ds.load()
    break

if refprevyr:
    reference_ds.to_netcdf(f'{reference_dir}reference_images_{year-1}-07-15_{year-1}-09-01.nc')
else:
    reference_ds.to_netcdf(f'{reference_dir}reference_images_{year}-07-15_{year}-09-01.nc')
