# load_data.py
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy


def intersect_rectilinear(xlat,xlon,obs_lat,obs_lon):
    """
    Intersects lat and lon location of insitu observation with SnowModel grid
    Input:
      xlat - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      xlon - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      obs_lat - python float of latitude for observation.
      obs_lon - python float of longitude for observation.

    Output:
      idy - integer representing y-index of intersected point on SnowModel grid.
      idx - integer representing x-index of intersected point on SnowModel grid.
    """
    dist = np.sqrt((xlat - obs_lat)**2)
    idy = np.argwhere(dist == np.min(dist))[0][0]
    dist = np.sqrt((xlon - obs_lon)**2)
    idx = np.argwhere(dist == np.min(dist))[0][0]

    return idy,idx

def sentinel_wy_intersect_pt(fpath: str,
                     wy: int,
                     pt_x: float,
                     pt_y: float,
                     clipSentinel: bool = False,
                     clip_geom: gpd.GeoDataFrame = None):
    """
    Loads sentinel data for single water year with options to clip
    Input:
      fpath - sentinel fpath.
      wy - water year.
      clipSentinel - boolean indicating clipping.
      clip_geom - clip shape.
    Output:
      dict - dictionary object of yaml items
    """
    fname = f's1_backscatter_{wy-1}_10_01_{wy}_10_01_40m.nc'
    sentinel_bbox = xr.load_dataset(fpath + fname)
    sentinel_bbox = sentinel_bbox.rename({'__xarray_dataarray_variable__':'bscatter'})
    if clipSentinel:
        sentinel_clip = sentinel_bbox.rio.clip(clip_geom.to_crs('EPSG:32611').buffer(40).geometry)
    idy_pt,idx_pt = intersect_rectilinear(sentinel_clip.y.values,sentinel_clip.x.values,pt_y,pt_x)
    return sentinel_clip,idy_pt,idx_pt

def load_multidirectory_snowmodel_output(base_dir: str,
                                         var: str = 'swed',
                                         convertMM: bool = True):
    """
    Loads snowmodel point data from multiple simulation directories.
    Input:
      fpath - sentinel fpath.
      wy - water year.
      clipSentinel - boolean indicating clipping.
      clip_geom - clip shape.
    Output:
      dict - dictionary object of yaml items
    """
    data = []
    for path in os.listdir(base_dir):
        if 'wy_' in path and "ml_test" not in path:
            print(path)
            try:
                fpath = f'{base_dir}{path}/netcdf/{var}.nc'
                da = xr.open_dataarray(fpath)
                data.append(da)
            except:
                for file in os.listdir(f'{base_dir}{path}/netcdf/'):
                    if var in file:
                        fpath = f'{base_dir}{path}/netcdf/{file}'
                        da = xr.open_dataarray(fpath)
                        data.append(da)
    da_all = xr.concat(data,dim = 'time').sortby('time')
    if convertMM:
        da_all *= 1000
    return da_all

def load_cues_OBS_SWE(fpath: str,
                      convertMM: bool = True):
    """
    Loads snowmodel point data from multiple simulation directories.
    Input:
      fpath - sentinel fpath.
      wy - water year.
      clipSentinel - boolean indicating clipping.
      clip_geom - clip shape.
    Output:
      dict - dictionary object of yaml items
    """
    df_obs1 = pd.read_csv(f'{fpath}')
    df_obs1['DateTime'] = pd.to_datetime(df_obs1['DateTime'])
    if convertMM:
        df_obs1['SWE'] = df_obs1['SWE_cm'] * 10
    else:
        df_obs1 = df_obs1.rename(columns = {'SWE_cm':'SWE'})

    df_new = df_obs1[['DateTime','water_year','SWE']]
    return df_new[df_new['DateTime'] < np.datetime64('2017-10-01')],df_new[df_new['DateTime'] >= np.datetime64('2017-10-01')]

def load_cues_OBS_SWE_melt(fpath: str,
                      convertMM: bool = True):
    """
    Loads snowmodel point data from multiple simulation directories.
    Input:
      fpath - sentinel fpath.
      wy - water year.
      clipSentinel - boolean indicating clipping.
      clip_geom - clip shape.
    Output:
      dict - dictionary object of yaml items
    """
    df_obs1 = pd.read_csv(f'{fpath}')
    df_obs1['datetime'] = pd.to_datetime(df_obs1['datetime'])

    return df_obs1

def load_sentinel_backscatter(sent_dir: str,
                  proj_gdf: gpd.GeoDataFrame,
                  wy: int = None
                  ):
    """
    Loads sentinel backscatter data and clips it to domain.
    Input:
      sent_dir - sentinel fpath.
      wy - water year.
      proj_gdf - projected shape.
    Output:
      dict - dictionary object of yaml items
    """
    if wy is None:
        output = []
        for file in sorted(os.listdir(sent_dir)):
            if '.nc' in file:
                sentinel_mammoth = xr.load_dataset(sent_dir + file)
                sentinel_cues = sentinel_mammoth.rio.clip(proj_gdf.geometry)
                output.append(sentinel_cues)
        return xr.concat(output,dim = 'time').rename({'__xarray_dataarray_variable__':'bscatter'})
    else:
      fname = f's1_backscatter_{wy-1}_10_01_{wy}_10_01_40m.nc'
      sentinel_mammoth = xr.load_dataset(sent_dir + fname)
      sentinel_mammoth = sentinel_mammoth.rename({'__xarray_dataarray_variable__':'bscatter'})

      sentinel_cues = sentinel_mammoth.rio.clip(proj_gdf.geometry)
    return sentinel_cues

def load_point_snowmodel_liq_temp_swe_roff(wy: int,
                   sm_output_dir: str,
                   idy_: int = 0,
                   idx_: int = 0):
    """
    Loads snowmodel point data from multiple simulation directories.
    Input:
      sent_dir - sentinel fpath.
      wy - water year.
      proj_gdf - projected shape.
    Output:
      dict - dictionary object of yaml items
    """
    
    snowmodel_dict = {
        2013:'wy_2013-2022',
        2014:'wy_2013-2022',
        2015:'wy_2013-2022',
        2016:'wy_2013-2022',
        2017:'wy_2013-2022',
        2018:'wy_2013-2022',
        2019:'wy_2013-2022',
        2020:'wy_2013-2022',
        2021:'wy_2013-2022',
        2022:'wy_2013-2022',
        2023:'wy_2023',
        2024:'wy_2024',
    
    }
    start_date = np.datetime64(f'{wy-1}-10-01')
    end_date = np.datetime64(f'{wy}-09-01')
    sm_nc_corrected_dir = f'{sm_output_dir}{snowmodel_dict[wy]}/netcdf/'


    # load snowmodel output
    sm_ml_corrected_liq = xr.load_dataset(f'{sm_nc_corrected_dir}/multilayer_liq.nc')
    sm_sl_corrected_smlt = xr.load_dataset(f'{sm_nc_corrected_dir}/smlt.nc')
    sm_sl_corrected_roff = xr.load_dataset(f'{sm_nc_corrected_dir}/roff.nc')
    sm_sl_corrected_temp = xr.load_dataset(f'{sm_nc_corrected_dir}/multilayer_temp.nc')
    sm_sl_corrected_swed = xr.load_dataset(f'{sm_nc_corrected_dir}/swed.nc')
    # slice time and space
    modeled_runoff_corrected = sm_sl_corrected_roff.where(sm_sl_corrected_roff.time >= start_date,drop = True).where(sm_sl_corrected_roff.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    modeled_temp_corrected = sm_sl_corrected_temp.where(sm_sl_corrected_temp.time >= start_date,drop = True).where(sm_sl_corrected_temp.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    modeled_liq_corrected = sm_ml_corrected_liq.where(sm_ml_corrected_liq.time >= start_date,drop = True).where(sm_ml_corrected_liq.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    modeled_swe_corrected = sm_sl_corrected_swed.where(sm_sl_corrected_swed.time >= start_date,drop = True).where(sm_sl_corrected_swed.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    # combine datarrays into dataset.
    ds_phase = copy.deepcopy(modeled_temp_corrected)
    ds_phase['melt'] = xr.DataArray(modeled_liq_corrected['melt'],dims = ['time','layer'])
    ds_phase['swed'] = xr.DataArray(modeled_swe_corrected['swed'],dims = 'time')
    ds_phase['roff'] = xr.DataArray(modeled_runoff_corrected['roff'],dims = 'time')
    ml_mask = (ds_phase['tsfc'] >= -9998.0)
    ds_phase['tsfc'] = ds_phase['tsfc'].where(ml_mask)
    ds_phase['melt'] = ds_phase['melt'].where(ml_mask)

    return ds_phase

def load_multifile_snowmodel_output(dir_: str,
                                    var: str
                                    ):
    """
    Docstring for load_multifile_snowmodel_output
    
    :param dir_: Description
    :type dir_: str
    :param var: Description
    :type var: str
    """
    if var == 'swed':
        file_lst = [dir_ + i for i in os.listdir(f'{dir_}') if (var in i) and ('multilayer' not in i)]
        file_lst = [i for i in file_lst if 'multilayer_swed' not in i]
    else:
        file_lst = [dir_ + i for i in os.listdir(f'{dir_}') if var in i]
    ds = xr.open_mfdataset(sorted(file_lst),concat_dim = 'time',combine = 'nested')
    return ds


def load_point_snowmodel_liq_temp_swe_roff_multifile(wy: int,
                   sm_output_dir: str,
                   idy_: int = 0,
                   idx_: int = 0):
    """
    Loads snowmodel point data from multiple simulation directories.
    Input:
      sent_dir - sentinel fpath.
      wy - water year.
      proj_gdf - projected shape.
    Output:
      dict - dictionary object of yaml items
    """
    
    snowmodel_dict = {
        2017:'wy_2017',
        2018:'wy_2018',
        2019:'wy_2019',
        2020:'wy_2020',
        # 2021:'wy_2013-2022',
        # 2022:'wy_2013-2022',
        # 2023:'wy_2023',
        # 2024:'wy_2024',
    
    }
    start_date = np.datetime64(f'{wy-1}-10-01')
    end_date = np.datetime64(f'{wy}-09-01')
    sm_nc_corrected_dir = f'{sm_output_dir}{snowmodel_dict[wy]}/netcdf/'

    print(sm_nc_corrected_dir)
    # load snowmodel output
    sm_ml_corrected_liq = load_multifile_snowmodel_output(sm_nc_corrected_dir,'multilayer_liq')
    print('Loaded multilayer liquid spatial domain.')
    # sm_sl_corrected_smlt = xr.load_dataset(f'{sm_nc_corrected_dir}/smlt.nc')
    sm_sl_corrected_roff = load_multifile_snowmodel_output(sm_nc_corrected_dir,'roff')
    print('Loaded runoff spatial domain.')
    sm_sl_corrected_temp = load_multifile_snowmodel_output(sm_nc_corrected_dir,'multilayer_temp')
    print('Loaded multilayer temperature spatial domain.')
    sm_sl_corrected_swed = load_multifile_snowmodel_output(sm_nc_corrected_dir,'swed')
    print('Loaded sinlge Layer SWE spatial domain.')
    sm_ml_corrected_swed = load_multifile_snowmodel_output(sm_nc_corrected_dir,'multilayer_swed')
    print('Loaded multilayer SWE spatial domain.')
    # slice time and space
    modeled_runoff_corrected = sm_sl_corrected_roff.where(sm_sl_corrected_roff.time >= start_date,drop = True).where(sm_sl_corrected_roff.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).load()
    print('Indexed runoff.')
    modeled_temp_corrected = sm_sl_corrected_temp.where(sm_sl_corrected_temp.time >= start_date,drop = True).where(sm_sl_corrected_temp.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).load()
    print('Indexed temperature.')
    modeled_liq_corrected = sm_ml_corrected_liq.where(sm_ml_corrected_liq.time >= start_date,drop = True).where(sm_ml_corrected_liq.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).load()
    print('Indexed liquid.')
    modeled_swe_corrected = sm_sl_corrected_swed.where(sm_sl_corrected_swed.time >= start_date,drop = True).where(sm_sl_corrected_swed.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).load()
    print('Indexed SWE.')
    modeled_ml_swe_corrected = sm_ml_corrected_swed.where(sm_ml_corrected_swed.time >= start_date,drop = True).where(sm_ml_corrected_swed.time <= end_date,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).load()
    print('Indexed ML SWE.')
    modeled_ml_swe_corrected = modeled_ml_swe_corrected.rename({"swed":"ml_swed"})
    # combine datarrays into dataset.
    ds_phase = copy.deepcopy(modeled_temp_corrected)
    ds_phase['melt'] = xr.DataArray(modeled_liq_corrected['melt'],dims = ['time','layer'])
    ds_phase['ml_swed'] = xr.DataArray(modeled_ml_swe_corrected['ml_swed'],dims = ['time','layer'])
    ds_phase['swed'] = xr.DataArray(modeled_swe_corrected['swed'],dims = 'time')
    ds_phase['roff'] = xr.DataArray(modeled_runoff_corrected['roff'],dims = 'time')
    ml_mask = (ds_phase['tsfc'] >= -9998.0)
    ds_phase['tsfc'] = ds_phase['tsfc'].where(ml_mask)
    ds_phase['melt'] = ds_phase['melt'].where(ml_mask)
    ds_phase['ml_swed'] = ds_phase['ml_swed'].where(ml_mask)

    return ds_phase

def load_multifile_snowmodel_output_2(dir_: str,
                                      var: str,
                                      idy_: int,
                                      idx_: int):
    """
    Like load_multifile_snowmodel_output but subsets to a single grid cell
    before loading into memory, avoiding reading the full spatial domain.
    """
    file_lst = sorted([dir_ + i for i in os.listdir(dir_) if var in i])
    ds = xr.open_mfdataset(file_lst, concat_dim='time', combine='nested')
    return ds.isel(south_north=idy_, east_west=idx_)


def load_point_snowmodel_liq_temp_swe_roff_multifile_2(wy: int,
                                                        sm_output_dir: str,
                                                        idy_: int = 0,
                                                        idx_: int = 0):
    """
    Efficient version of load_point_snowmodel_liq_temp_swe_roff_multifile.
    Subsets to the target grid cell before loading data into memory, and uses
    sel(time=slice(...)) instead of where() for time filtering.
    """
    snowmodel_dict = {
        2017: 'wy_2017',
        2018: 'wy_2018',
        2019: 'wy_2019',
        2020: 'wy_2020',
    }
    start_date = np.datetime64(f'{wy-1}-10-01')
    end_date = np.datetime64(f'{wy}-09-01')
    sm_nc_corrected_dir = f'{sm_output_dir}{snowmodel_dict[wy]}/netcdf/'

    print(sm_nc_corrected_dir)

    liq  = load_multifile_snowmodel_output_2(sm_nc_corrected_dir, 'multilayer_liq',  idy_, idx_)
    print('Loaded multilayer liquid.')
    roff = load_multifile_snowmodel_output_2(sm_nc_corrected_dir, 'roff',            idy_, idx_)
    print('Loaded runoff.')
    temp = load_multifile_snowmodel_output_2(sm_nc_corrected_dir, 'multilayer_temp', idy_, idx_)
    print('Loaded temp.')
    swed = load_multifile_snowmodel_output_2(sm_nc_corrected_dir, 'swed',            idy_, idx_)
    print('Loaded swed.')

    time_slice = slice(start_date, end_date)
    liq  = liq.sel(time=time_slice)
    roff = roff.sel(time=time_slice)
    temp = temp.sel(time=time_slice)
    swed = swed.sel(time=time_slice)

    # merge lazy arrays into one dataset, then load in a single pass
    ds_phase = xr.merge([temp, liq[['melt']], swed[['swed']], roff[['roff']]]).load()
    print('Loaded point data.')

    ml_mask = ds_phase['tsfc'] >= -9998.0
    ds_phase['tsfc'] = ds_phase['tsfc'].where(ml_mask)
    ds_phase['melt'] = ds_phase['melt'].where(ml_mask)

    return ds_phase


def load_sentinel_reference(directory: str,
                            shape_crs: str,
                            clip_shape: gpd.GeoDataFrame = None):
    """
    Loads Sentinel reference backscatter.
    Input:
      directory - sentinel reference scene directory.
      clip_shape - clip shape.
    Output:
      dict - dictionary object of yaml items
    """
    files = [directory + i for i in os.listdir(directory)]
    if len(files) > 1:
        print('MULTIPLE FILES IN DIRECTORY!!!')
        print(files)
        return
    else:
        ds_ref = xr.load_dataset(files[0])
        ds_ref = ds_ref.rio.write_crs(shape_crs)
        if clip_shape is None:
            return ds_ref
        else:
            return ds_ref.rio.clip(clip_shape.geometry)
      


def fill(x,y1,y2,color_var,vmin_,vmax_,cmap_ = 'inferno'):
    if np.isnan(y1).sum() != np.isnan(y2).sum():
        mask = np.isnan(y2)
        y2[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y2[~mask])
        
        mask = np.isnan(color_var)
        color_var[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), color_var[~mask])
    polygon = plt.fill_between(x, y1, y2, lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    gradient = plt.imshow(color_var.reshape(1, -1), cmap=cmap_, aspect='auto',vmax = vmax_,vmin = vmin_,
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    return

def visualize_3D_sm_var(y_filevar,color_filevar,netcdf_fpath,idy_,idx_,
                        vmin_ = -15,vmax_ = 0, cmap_ = 'inferno',showLayers = False,
                        start_date = None, end_date = None,plotLWC = True,showPlot = True,
                        saveFig = False,fpath = None):
    """
      Visualization of multilayer output of SnowModel as time series at one grid cell.
      Input:
        y_filevar - python string of SnowModel variable to visualize as layers in y-dimension.
                Usually either 'multilayer_snod' or multilayer_swed.
        color_filevar - python string of SnowModel variable to plot as imshow in between layers.
                Examples include 'multilayer_sden', or multilayer_temp'.
        netcdf_fpath - relative file path to directory with SnowModel netcdf output.
        idy_ - python integer for y-dimension of grid cell.
        idx_ - python integer for x-dimension of grid cell.
        vmin_ - python float of colorbar min.
        vmax_ - python float of colorbar max.
        cmap_ - python string for colormap.
        showLayers - boolean for showing layers in plot.
        start_date - numpy datetime64 object for xmin plotting purposes.
        end_date - numpy datetime64 object for xmax plotting purposes.
      Output:
        None
    """
    ## based on filename get xarray variable name.
    y_varname = get_xarray_var(y_filevar)
    color_varname = get_xarray_var(color_filevar)

    ## multilayer depth or SWE ##
    file_lst = [netcdf_fpath + i for i in os.listdir(f'{netcdf_fpath}') if y_filevar in i]
    print(file_lst)
    ds = xr.open_mfdataset(sorted(file_lst),concat_dim = 'time',combine = 'nested')
    da = ds[y_varname].dropna(dim = 'layer',how = 'all')

    ## color variable ##
    file_lst = [netcdf_fpath + i for i in os.listdir(f'{netcdf_fpath}') if color_filevar in i]
    ds_ml_t = xr.open_mfdataset(sorted(file_lst),concat_dim = 'time',combine = 'nested')
    da_ml_t = ds_ml_t.where(ds_ml_t[color_varname] > -9998.0)[color_varname]
    da_ml_t = da_ml_t.dropna(dim = 'layer',how = 'all')

    if plotLWC:
        fc_param = 50.0
        Tf = 273.16
        lwc = 1.0/(1.0+((Tf - (np.minimum(da_ml_t.values+273.15,Tf)))**2))
        da_ml_t = xr.DataArray(
            data=lwc,
            dims=["time", "layer", "south_north","east_west"],
            coords=dict(
                time=(["time"], da_ml_t.time.values),
                layer=(["layer"], da_ml_t.layer.values),
                xlat=(["south_north", "east_west"], da_ml_t.xlat.values),
                xlon=(["south_north", "east_west"], da_ml_t.xlon.values),
            ),
        )
        da_ml_t.rename('tsfc')

    ds_new = da[:,:,idy_,idx_].sum(dim = 'layer') - da[:,::-1,idy_,idx_].cumsum(dim = 'layer',skipna = True)

    if showPlot:
        ## plotting ##
        fig,ax = plt.subplots(dpi = 200)
        da[:,:,idy_,idx_].sum(dim = 'layer').plot(ax=ax,label = f'total depth',color = 'black')
        xlim = plt.xlim()
        ylim = plt.ylim()

        ## plot top layer ##
        fill(ds_new.where(ds_new >= 0)[:,0].time.values,ds_new.where(ds_new >= 0)[:,0].values,
            da[:,:,idy_,idx_].sum(dim = 'layer').values,da_ml_t[:,-1,idy_,idx_].values,vmin_,vmax_,cmap_)
        if showLayers:
            ds_new.where(ds_new >= 0)[:,0].plot(ax=ax)

        ## loop through remaining layers ##
        for i in range(1,ds_new.shape[1]):
            fill(ds_new.where(ds_new >= 0)[:,i].time.values,ds_new.where(ds_new >= 0)[:,i-1].values,
                ds_new.where(ds_new >= 0)[:,i].values,da_ml_t[:,-i-1,idy_,idx_].values,vmin_,vmax_,cmap_)
        if showLayers:
            ds_new.where(ds_new >= 0)[:,i].plot(ax=ax)


        plt.colorbar()
        plt.xlim(xlim)
        plt.ylim(ylim)
        if (start_date is not None) and (end_date is not None):
            ax.set_xlim([start_date,end_date])
        if saveFig:
            plt.savefig(f'{fpath}.png',dpi = 300)
        plt.show()
    
    return da_ml_t,da,ds_new

def get_xarray_var(file_varname):
    """
      Get xarray variable name based on filename.
      Input:
        file_varname - python string of SnowModel variable to visualize as layers in y-dimension.
                Usually either 'multilayer_snod' or multilayer_swed.
      Output:
        xarray_varname - python string of xarray variable name.
    """
    if file_varname == 'multilayer_snod':
        xarray_varname = 'snod'
    elif file_varname == 'multilayer_swed':
        xarray_varname = 'swed'
    elif file_varname == 'multilayer_sden':
        xarray_varname = 'sden'
    elif file_varname == 'multilayer_temp':
        xarray_varname = 'tsfc'
    elif file_varname == 'multilayer_flux':
        xarray_varname = 'flux'
    elif file_varname == 'multilayer_diam': # this is wrong.
        xarray_varname = 'snod'
    elif file_varname == 'multilayer_cond':
        xarray_varname = 'cond'

    return xarray_varname

def sm_intersect_curvilinear(xlat,xlon,obs_lat,obs_lon):
    """
    Intersects lat and lon location of insitu observation with SnowModel grid
    Input:
      xlat - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      xlon - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      obs_lat - python float of latitude for observation.
      obs_lon - python float of longitude for observation.

    Output:
      idy - integer representing y-index of intersected point on SnowModel grid.
      idx - integer representing x-index of intersected point on SnowModel grid.
    """
    dist = np.sqrt((xlat - obs_lat)**2 + (xlon - obs_lon)**2)
    idy,idx = np.argwhere(dist == np.min(dist))[0]
    return idy,idx

def sm_intersect_rectilinear(xlat,xlon,obs_lat,obs_lon):
    """
    Intersects lat and lon location of insitu observation with SnowModel grid
    Input:
      xlat - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      xlon - 2D numpy array representing latitude values for each SnowModel grid
                      point.
      obs_lat - python float of latitude for observation.
      obs_lon - python float of longitude for observation.

    Output:
      idy - integer representing y-index of intersected point on SnowModel grid.
      idx - integer representing x-index of intersected point on SnowModel grid.
    """
    dist = np.sqrt((xlat - obs_lat)**2)
    idy = np.argwhere(dist == np.min(dist))[0][0]
    dist = np.sqrt((xlon - obs_lon)**2)
    idx = np.argwhere(dist == np.min(dist))[0][0]

    return idy,idx
    