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

# # -- inputs.

start_date = sys.argv[1]
end_date = sys.argv[2]
resolution = int(sys.argv[3])
domain = sys.argv[4]

year = int(end_date[0:4])

print(f'domain {domain}; processing {year}; resolution {resolution}')

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
sentinel_cues_10m_fpath = metadata.get_sentinel_fpath(cues_cfg,'10m')
sentinel_cues_40m_fpath = metadata.get_sentinel_fpath(cues_cfg,'40m')
# load snowmodel filepath
sm_cues_OBS_swe_fpath = metadata.get_cuesOBS_fpath(cues_cfg,'swe')
sm_cues_OBS_sweMelt_fpath = metadata.get_cuesOBS_fpath(cues_cfg,'swe_melt')

## --------------- KSP
ksp_cfg = metadata.load_yaml(Path(f"../../../../configs/KSP.yaml"))
ksp_lat,ksp_lon,ksp_x,ksp_y,ksp_crs = metadata.get_point_geography(ksp_cfg,'ksp')

## --------------- KSP
tuolumne_cfg = metadata.load_yaml(Path(f"../../../../configs/Tuolumne.yaml"))
tuolumneGauge_lat,tuolumneGauge_lon,tuolumneGauge_x,tuolumneGauge_y,tuolumneGauge_crs = metadata.get_point_geography(tuolumne_cfg,'tuolumne_gauge')
lyellGauge_lat,lyellGauge_lon,lyellGauge_x,lyellGauge_y,lyellGauge_crs = metadata.get_point_geography(tuolumne_cfg,'lyell_gauge')
danaGauge_lat,danaGauge_lon,danaGauge_x,danaGauge_y,danaGauge_crs = metadata.get_point_geography(tuolumne_cfg,'dana_gauge')
tuolumnePillow_lat,tuolumnePillow_lon,tuolumnePillow_x,tuolumnePillow_y,tuolumnePillow_crs = metadata.get_point_geography(tuolumne_cfg,'tuolumne_pillow')
danaPillow_lat,danaPillow_lon,danaPillow_x,danaPillow_y,danaPillow_crs = metadata.get_point_geography(tuolumne_cfg,'dana_pillow')

# cues
mammoth_geog_fpath,aso_crs = metadata.get_shapefile_fpath(cues_cfg,"shapefile_fpath","zoom_1")
cues_geog_fpath,aso_crs = metadata.get_shapefile_fpath(cues_cfg,"shapefile_fpath","zoom_2")
# ksp
ksp_zoom1_geog_fpath,aso_crs = metadata.get_shapefile_fpath(ksp_cfg,"shapefile_fpath","zoom_1")
ksp_zoom2_geog_fpath,aso_crs = metadata.get_shapefile_fpath(ksp_cfg,"shapefile_fpath","zoom_2")
# tuolumne
tuolumne_geog_fpath,aso_crs = metadata.get_shapefile_fpath(tuolumne_cfg,"shapefile_fpath","zoom_1")
dana_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'dana_watershed')
lyell_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'lyell_watershed')
tuolumne_watershed_fpath = metadata.get_watershed_fpath(tuolumne_cfg,'tuolumne_watershed')
# ASO
aso_cfg = metadata.load_yaml(Path(f"../../../../configs/ASO.yaml"))
uscatm_geog_fpath,aso_crs = metadata.get_shapefile_fpath(aso_cfg,"basin_shapefile","USCATM")
uscasj_geog_fpath,aso_crs = metadata.get_shapefile_fpath(aso_cfg,"basin_shapefile","USCASJ") 
# read shapefile
## mammoth
mammoth_geog_gdf = gpd.read_file(mammoth_geog_fpath)
mammoth_proj_gdf = mammoth_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## cues
cues_geog_gdf = gpd.read_file(cues_geog_fpath)
cues_proj_gdf = cues_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## mammoth
ksp_zoom1_geog_gdf = gpd.read_file(ksp_zoom1_geog_fpath)
ksp_zoom1_proj_gdf = ksp_zoom1_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## cues
ksp_zoom2_geog_gdf = gpd.read_file(ksp_zoom2_geog_fpath)
ksp_zoom2_proj_gdf = ksp_zoom2_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## uscatm
uscatm_geog_gdf = gpd.read_file(uscatm_geog_fpath)
uscatm_proj_gdf = uscatm_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## uscatm
uscasj_geog_gdf = gpd.read_file(uscasj_geog_fpath)
uscasj_proj_gdf = uscasj_geog_gdf.to_crs(f'EPSG:{aso_crs}')
## tuolumne
tuolumne_geog_gdf = gpd.read_file(tuolumne_geog_fpath)
tuolumne_proj_gdf = tuolumne_geog_gdf.to_crs(f'EPSG:{aso_crs}')


if domain == 'cues':
    bbox_gdf = mammoth_geog_gdf
    bbox_gdf.attrs = {'filename':mammoth_geog_fpath}
    geojson = 'mammoth.json'
elif domain == 'tuolumne':
    bbox_gdf = tuolumne_geog_gdf
    bbox_gdf.attrs = {'filename':tuolumne_geog_fpath}
    geojson = 'tuolumne.json'
else:
    print('Passed incorrect domain! Please resubmit!')
    sys.exit()
save_dir = f'/glade/derecho/scratch/rossamower/snow/snowmodel/school/uw/cues/data/sentinel/{domain}/{resolution}_m/{year}/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
reference_scenes_aggregation_technique = 'median' # mean / median / max

time_reference_scenes = slice(f'{year}-07-15',f'{year}-09-01') 


# # download data

## straight backscatter
sentinel_ds = create_dataset.get_s1_rtc(bbox_gdf,
                          start_date,
                          end_date,
                          resolution)

save_dir_bscatter = f'{save_dir}bscatter/'
save_dir_meltThresh = f'{save_dir}melt_thresh/'

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
