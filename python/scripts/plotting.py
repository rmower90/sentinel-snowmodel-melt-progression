# plotting.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextily as ctx
import numpy as np
from matplotlib import cm, colors
import geopandas as gpd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import os
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.dates as mdates


def get_orbit_state(sentinel_slice, t_idx):
    """
    Try common places this metadata might live.
    Returns 'ascending' / 'descending' (lowercase) or None.
    """
    # 1) In the dataarray attrs of the band/time slice
    try:
        da = 10 * np.log10(sentinel_slice.bscatter[t_idx, 0, :, :])
        val = da.attrs.get("sat:orbit_state", None)
        if isinstance(val, (str, np.str_)):
            return val.lower()
    except Exception:
        pass

    # 2) In the parent dataset attrs for this time slice
    try:
        val = sentinel_slice.isel(time=t_idx).attrs.get("sat:orbit_state", None)
        if isinstance(val, (str, np.str_)):
            return val.lower()
    except Exception:
        pass

    # 3) As a variable/coordinate on the dataset (per-time)
    for key in ["sat:orbit_state", "sat_orbit_state", "orbit_state"]:
        if key in sentinel_slice and "time" in sentinel_slice[key].dims:
            try:
                v = sentinel_slice[key].isel(time=t_idx).item()
                if isinstance(v, (str, np.str_)):
                    return v.lower()
            except Exception:
                pass

    return None  # unknown


def subplot_sentinel_spatial(polarization: str,
                            sentinel_ds: xr.DataArray,
                            month_start: int,
                            month_end: int,
                            geom_proj: gpd.GeoDataFrame,
                            point_idx: int,
                            point_idy: int,
                            domain: str = 'CUES',
                            showOrbitState: bool = True):
    """
    Spatial subplots of given polarization over a sequence of months.
    Input:
      polarization - string of polarization ("vv","vh")
      sentinel_ds - sentinel xarray dataset.
      month_start - integer for start month.
      month_end - integer for end month.
      geom_proj - subdomain geometry.
      point_idx - point x-coordinate.
      point_idx - point y-coordinate.
      domain - subdomain string.
    Output:
      std_arr - numpy array for mean standard deviation.
    """
    
    if showOrbitState:
        # map orbit -> border color
        orbit_colors = {
            "ascending": "tab:blue",
            "descending": "tab:orange"
        }
    sentinel_slice = (
        sentinel_ds
        .where(sentinel_ds.time.dt.month >= 2, drop=True)
        .where(sentinel_ds.time.dt.month <= 6, drop=True)
        .where(sentinel_ds.band == polarization, drop=True)
    )

    # ---- layout params ----
    n_cols = 10
    n_times = sentinel_slice.time.shape[0]
    n_rows = (n_times + n_cols - 1) // n_cols  # ceil division

    # use constrained layout so the colorbar has space
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2), constrained_layout=True)
    ax = np.atleast_2d(ax)  # ensure 2D

    # ---- shared color scaling ----
    vmin, vmax = -25, -5
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    std_lst = []
    for t_idx in range(n_times):
        r, c = divmod(t_idx, n_cols)
        da = 10 * np.log10(sentinel_slice.bscatter[t_idx, 0, :, :])
        std = np.nanstd(da.values)
        std_lst.append(std)

        # plot without colorbar, but with the shared norm/cmap
        da.plot(ax=ax[r, c], alpha=0.7, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
        geom_proj.plot(ax=ax[r, c], color='none', edgecolor='red')
        ax[r, c].scatter(sentinel_slice.x[point_idx], sentinel_slice.y[point_idy], color='red', s=5)
        ax[r, c].set_axis_off()
        ax[r, c].set_title(f"{str(sentinel_slice.time[t_idx].values)[:10]}\nstd:{std:.1f}", fontweight='bold', fontsize=9)

        # ---- add colored border for orbit state ----
        if showOrbitState:
            orbit = get_orbit_state(sentinel_slice, t_idx)
            edgecolor = orbit_colors.get(orbit, "0.5")  # gray if unknown
            rect = Rectangle(
                (0, 0), 1, 1,
                transform=ax[r, c].transAxes,
                fill=False,
                linewidth=3,
                edgecolor=edgecolor,
                joinstyle='round',
                zorder=10
            )
            ax[r, c].add_patch(rect)


    # turn off any leftover axes (if grid not perfectly filled)
    total_slots = n_rows * n_cols
    for blank_idx in range(n_times, total_slots):
        r, c = divmod(blank_idx, n_cols)
        ax[r, c].set_axis_off()

    std_arr = np.array(std_lst)

    # ---- single colorbar on the right, in its own axes ----
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig.canvas.draw()  # after subplots exist so positions are known
    grid_bbox = Bbox.union([a.get_position() for a in ax.ravel()])  # bounds of subplot grid

    pad  = 0.012   # gap between grid and colorbar
    lift = 0.09    # vertical space to reserve at bottom for legend (tweak)
    cbar_w = 0.018 # colorbar width in figure coords

    cax = fig.add_axes([
        grid_bbox.x1 + pad,           # left
        grid_bbox.y0 + lift,          # bottom (lifted)
        cbar_w,                       # width
        grid_bbox.height - (lift+0.02)  # height (shorter to match the lift)
    ])
    cbar = fig.colorbar(mappable, cax=cax, label='Backscatter (dB)')

    # orbit legend under the colorbar
    if showOrbitState:
        legend_elems = [
            Line2D([0], [0], color="tab:blue",   lw=3, label="Ascending"),
            Line2D([0], [0], color="tab:orange", lw=3, label="Descending"),
            Line2D([0], [0], color="0.5",        lw=3, label="Unknown"),
        ]
        fig.legend(
            handles=legend_elems,
            loc="lower right",
            bbox_to_anchor=(0.992, 0.02),  # tweak as needed
            frameon=False,
            title="Orbit",
        )
    

    plt.suptitle(f'{domain} - {polarization.upper()}; avg std: {np.nanmean(std_arr):.2f}', fontweight='bold', fontsize=20)
    plt.show()

    return std_arr


def model_sar_obs_cues_timeseries(sm_cues_corrected_fpath: str,
                                  cues_obs_swe_df: pd.DataFrame,
                                  df_lys: pd.DataFrame,
                                  sentinel_cues: xr.DataArray,
                                  start_date: np.ndarray,
                                  end_date: np.ndarray,
                                  idy_cues: int,
                                  idx_cues: int,
                                  idy_: int = 0,
                                  idx_: int = 0,
                                  showLayers: bool = False,
                                  vmin_: int = -15,
                                  vmax_:int = 0,
                                  cmap_ = 'inferno',
                                  is3x3Mean: bool = False,
                                  ):
    """
    time series of data at cues.
    Input:
      polarization - string of polarization ("vv","vh")
      sentinel_ds - sentinel xarray dataset.
      month_start - integer for start month.
      month_end - integer for end month.
      geom_proj - subdomain geometry.
      point_idx - point x-coordinate.
      point_idx - point y-coordinate.
      domain - subdomain string.
    Output:
      std_arr - numpy array for mean standard deviation.
    """
    wy = int(str(end_date)[0:4])
    
    color_dict = {64:'C0',
                 137:'C1',
                 144:'C2'
                 }
    
    swe_ymax = (cues_obs_swe_df[(cues_obs_swe_df['DateTime'] >= start_date) & (cues_obs_swe_df['DateTime'] <= end_date)]['SWE'].max() / 1000) * 1.4
    
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

    sm_nc_dir = f'{sm_cues_corrected_fpath}{snowmodel_dict[wy]}/netcdf/'

    temp_ml_ds, swed_top, swed_ml_ds = visualize_3D_sm_var('multilayer_swed',
                            'multilayer_temp',
                            sm_nc_dir,
                            0,
                            0,
                           start_date = start_date,
                           end_date = end_date,
                           showPlot = False,
                           plotLWC = False,
                           )
    
    fig = plt.figure(figsize=(12, 11))  # Taller figure to fit all plots

    gs = gridspec.GridSpec(
        4, 2,  # 4 rows, 2 columns
        width_ratios=[20, 1],
        height_ratios=[3, 1, 1, 2],  # main plot taller
        wspace=0.05,
        hspace=0.3
    )

    # Main axes (top)
    ax0 = fig.add_subplot(gs[0, 0])

    # Additional axes below
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    # ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    # Colorbar axes next to the top plot
    cax = fig.add_subplot(gs[0, 1])

    # 4️⃣ Main time series
    swed_top[:,:,idy_,idx_].sum(dim='layer').plot(ax=ax0, label='Modeled', color='black')

    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()

    # 5️⃣ Top layer gradient fill
    gradient = fill(
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].time.values,
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].values,
        swed_top[:,:,idy_,idx_].sum(dim='layer').values,
        temp_ml_ds[:,-1,idy_,idx_].values,
        vmin_,
        vmax_,
        axes=ax0,
        cmap_=cmap_
    )

# Remaining layers
    for i in range(1, swed_ml_ds.shape[1]):
        gradient = fill(
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].time.values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i-1].values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].values,
            temp_ml_ds[:,-i-1,idy_,idx_].values,
            vmin_,
            vmax_,
            axes=ax0,
            cmap_=cmap_
    )

    # Colorbar
    cbar = fig.colorbar(gradient, cax=cax)
    cbar.set_label("Temperature (°C)")

    # Observations
    ax0.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'Observed')
    ax0.legend()
    # Axes formatting
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_ylim(0,swe_ymax)
    ax0.grid(True, linestyle='--')
    if (start_date is not None) and (end_date is not None):
        ax0.set_xlim([start_date, end_date])
    ax0.set_title("Modeled and Observed SWE",fontweight = 'bold')
    ax0.set_ylabel("SWE\n[m]",fontweight = 'bold')
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_xlabel("")

    # 6️⃣ Example content for ax1
    # Replace this with your actual data plotting in ax1
    # Here is an example bar plot with a legend
    # df_lys = df_lys.reset_index()
    df_lys_ = df_lys[(df_lys['datetime'] >= start_date) & (df_lys['datetime'] <= end_date)]
    dates = df_lys_['datetime']
    # ax1.bar(dates, np.random.randint(0, 100, size=len(dates)), width=1, label="Example Series")

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb1_south", "tb2_south", "tb3_south", "tb4_south"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax1.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax1.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax1.grid(True, linestyle="--")
    ax1.set_title("Observed Southern Lysimeters",fontweight = 'bold')
    ax1.tick_params(axis='x', labelbottom=False)

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb5_north", "tb6_north", "tb7_north", "tb8_north"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax2.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax2.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax2.grid(True, linestyle="--")
    ax2.set_title("Observed Northern Lysimeters",fontweight = 'bold')
    ax2.tick_params(axis='x', labelbottom=False)

    sentinel_cues_yr = sentinel_cues.where(sentinel_cues.time >= start_date,drop = True)
    sentinel_cues_yr = sentinel_cues_yr.where(sentinel_cues_yr.time <= end_date,drop = True)
    tmin1 = sentinel_cues_yr.time[-1].values
    tmax1 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date1 = sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
    
        if is3x3Mean:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idx_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
    
        ax3.plot(date1,vals1,label = f'{orbit}, vv',color = color_dict[orbit],marker = '*')



    tmin2 = sentinel_cues_yr.time[-1].values
    tmax2 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date2 = sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        if is3x3Mean:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idy_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
        ax3.plot(date2,vals2,label = f'{orbit}, vh',color = color_dict[orbit],linestyle = '--',marker = '*')

    ax3.set_ylabel("Backscatter\n[dB]",fontweight = 'bold')
    ax3.set_title("Sentinel-2 VV/VH",fontweight = 'bold')
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.grid(True, linestyle="--")

    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax1.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax2.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    ax3.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )


    plt.tight_layout()
    plt.show()  
    
    return


def plot_sentinel_melt_phases_continuous(melt_da, 
                                         ds_phase,
                                        ax = None):
    """
    melt_da : original melt DataArray (sat:relative_orbit, Time)
    ds_phase : output of classify_sentinel_melt_phases
               (must have phase_code(Time,))
    """
    PHASE_CODES = {
        # 0: "dry",
        3: "Output",
        2: "Ripening",
        1: "Moistening",
    }
    time = ds_phase["Time"].values
    phase_code = ds_phase["phase_code"].values

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 5),constrained_layout=True)
    else:
        ax1 = ax


    # -------- bottom panel: continuous phase bands --------
    colors = {
        # 0: "None",   # dry
        1: "lightblue",    # moistening
        2: "palegoldenrod",  # ripening
        3: "lightcoral",     # output
    }

    # midpoints between time stamps (use timedeltas!)
    if time.size > 1:
        dt = time[1:] - time[:-1]               # timedelta64
        mid = time[:-1] + dt / 2
    else:
        mid = time.copy()

    # boundaries for each block: [b[i], b[i+1]) has phase phase_code[i]
    b = np.empty(time.size + 1, dtype="datetime64[ns]")

    if time.size > 1:
        first_step = time[1] - time[0]
        last_step  = time[-1] - time[-2]
        b[1:-1] = mid
        b[0] = time[0] - first_step / 2
        b[-1] = time[-1] + last_step / 2
    else:
        # single time stamp: just make a tiny interval around it
        half = np.timedelta64(12, "h")
        b[0] = time[0] - half
        b[1] = time[0] + half

    # draw one rectangle per time step; no gaps
    for i in range(time.size):
        code = int(phase_code[i])
        ax1.axvspan(
            b[i], b[i+1],
            facecolor=colors.get(code, "white"),
            alpha=0.5,
            edgecolor="none",
        )

    # build legend manually (unique labels)
    handles = []
    for code, name in PHASE_CODES.items():
        h = ax1.axvspan(
            b[0], b[0], facecolor=colors[code], alpha=0.5, edgecolor="none"
        )
        handles.append(h)
    # ax1.legend(handles, PHASE_CODES.values(), loc="upper right", fontsize=8, ncol=1)

    # ax1.set_yticks([])
    # ax1.set_ylim(0, 1)
    # ax1.set_xlabel("Time")
    # ax1.set_title("Sentinel-1 melt phases")

    if ax is None: 
        return fig, ax1
    else:
        return ax1,(handles,PHASE_CODES.values())

def plot_snowmelt_phases_filltop(
    ds,
    ds_phase,
    swe_var="swed",
    runoff_var="roff",
    title=None,
    t_start=None,
    t_end=None,
    figsize=(14, 4),
    ax = None,
    ylim_= None,
):
    """
    Plot SWE and runoff with phases shaded from the SWE line up to
    the top of the axis using fill_between.

    Phases (from classifier):
      0 = dry (not shaded)
      1 = moistening  (light blue)
      2 = ripening    (light yellow)
      3 = output      (light red)
    """

    # --- time selection / zoom ---
    if t_start is not None or t_end is not None:
        tsel = slice(t_start, t_end)
        ds_ = ds.sel(time=tsel)
        ph_ = ds_phase.sel(time=tsel)
    else:
        ds_ = ds
        ph_ = ds_phase

    time   = ds_["time"]
    swe    = ds_[swe_var]
    if runoff_var is not None:
        runoff = ds_[runoff_var]
    else:
        runoff = None

    phase_code = ph_["phase_code_3h"]

    # Boolean masks by phase code
    is_output    = (phase_code == 3)
    is_ripening  = (phase_code == 2)
    is_moist     = (phase_code == 1)

    # Convert to numpy for fill_between
    t_vals   = time.values
    swe_vals = swe.values.astype(float)

    # --- plotting ---
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax

    # SWE and runoff
    ax1.plot(time, swe, "k-", linewidth=1.6, label="SWE (m)")
    # ax2.plot(time, runoff, "b--", alpha=0.5, label="Runoff")

    # Decide top of shading: a bit above max SWE
    if ylim_ is not None:
        y_top = ylim_
    else:
        swe_max = np.nanmax(swe_vals)
        margin  = max(0.1, 0.2 * swe_max)  # at least 0.02 m, or 5%
        y_top   = swe_max + margin
    if ylim_ is None:
        ax1.set_ylim(bottom=0.0, top=y_top)
    else:
        ax1.set_ylim(bottom=0.0, top=ylim_)

    # Helper: shade a phase from SWE to y_top where mask is True
    def _shade_phase(mask_da, color, alpha):
        m = mask_da.values.astype(bool)
        if m.size == 0:
            return
        # Also require SWE to be finite
        valid = m & np.isfinite(swe_vals)
        if not np.any(valid):
            return
        ax1.fill_between(
            t_vals,
            swe_vals,
            y_top,
            where=valid,
            color=color,
            alpha=alpha,
            interpolate=True,
        )

    # Draw in any order (phases are mutually exclusive), but following priority:
    # moistening, ripening, output (last drawn is highest priority if ever overlap)
    _shade_phase(is_moist,    color="lightblue",     alpha=0.4)
    _shade_phase(is_ripening, color="palegoldenrod", alpha=0.5)
    _shade_phase(is_output,   color="lightcoral",    alpha=0.4)

    # Labels, grid, title
    ax1.set_ylabel("SWE (m)")
    # ax2.set_ylabel("Runoff")
    if title is not None:
        ax1.set_title(title)
    ax1.grid(alpha=0.3)
    if ax is None: fig.autofmt_xdate()

    # Build legend: lines + phase patches
    line_handles, line_labels = ax1.get_legend_handles_labels()
    # line_handles2, line_labels2 = ax2.get_legend_handles_labels()

    phase_patches = [
        Patch(facecolor="lightcoral",    alpha=0.4, label="Output"),
        Patch(facecolor="palegoldenrod", alpha=0.5, label="Ripening"),
        Patch(facecolor="lightblue",     alpha=0.4, label="Moistening"),
    ]

    # handles = line_handles + line_handles2 + phase_patches
    # labels  = line_labels  + line_labels2  + [p.get_label() for p in phase_patches]

    handles = phase_patches
    labels  = [p.get_label() for p in phase_patches]

    # remove duplicate labels
    uniq = dict(zip(labels, handles))
    if ax is None:
        ax1.legend(uniq.values(), uniq.keys(), loc="lower right")

    if ax is None: plt.tight_layout()
    if ax is None: plt.show()
    if ax is None: 
        return
    else:
        return uniq,y_top


def plot_melt_phases_intervals_filltop(
    ds,
    ds_phase,
    swe_var="swed",
    runoff_var="roff",
    title=None,
    t_start=None,
    t_end=None,
    figsize=(14, 4),
    ax=None,
    ylim_=None,
):
    """
    Plot SWE with Sentinel interval-based melt phases shaded from the SWE line 
    up to the top of the axis using fill_between.

    Phases (from Sentinel SAR):
      0 = dry (not shaded)
      1 = moistening  (light blue)
      2 = ripening    (light yellow)
      3 = output      (light red)

    Parameters:
    -----------
    ds : xarray.Dataset
        Time series data with variables: swe_var, runoff_var, time dimension
    ds_phase : xarray.Dataset
        Interval-based phases with dimensions: 'interval'
        Variables: 'interval_start', 'interval_end', 'phase_id', 'phase_name'
    swe_var : str
        Variable name for snow water equivalent (default: "swed")
    runoff_var : str
        Variable name for runoff (default: "roff")
    title : str, optional
        Plot title
    t_start, t_end : str, optional
        Time range for x-axis (ISO format, e.g., "2017-01-01")
    figsize : tuple
        Figure size (default: (14, 4))
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    ylim_ : float, optional
        Y-axis upper limit
    """
    import pandas as pd
    import xarray as xr

    # --- time selection / zoom ---
    if t_start is not None or t_end is not None:
        tsel = slice(t_start, t_end)
        ds_ = ds.sel(time=tsel)
    else:
        ds_ = ds

    time = ds_["time"]
    swe = ds_[swe_var]

    # Convert to numpy for plotting
    t_vals = time.values
    swe_vals = swe.values.astype(float)

    # --- Build phase_code array from intervals ---
    # Create a phase_code DataArray aligned with time dimension
    phase_code = xr.DataArray(
        np.zeros(len(time), dtype=int),
        coords={"time": time},
        dims=["time"],
    )

    # Map intervals to phase codes on the time grid
    for idx in range(len(ds_phase["interval"])):
        t_start_val = pd.Timestamp(ds_phase["interval_start"].values[idx])
        t_end_val = pd.Timestamp(ds_phase["interval_end"].values[idx])
        phase_id = int(ds_phase["phase_id"].values[idx])

        # Find indices within this interval
        mask = (time >= t_start_val) & (time <= t_end_val)
        phase_code.values[mask.values] = phase_id

    # Create boolean masks by phase code
    is_output = (phase_code == 3)
    is_ripening = (phase_code == 2)
    is_moist = (phase_code == 1)

    # --- plotting ---
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax
        fig = None

    # SWE time series
    ax1.plot(time, swe, color = 'brown', linestyle = '-', linewidth=1.6, label="SWE (m)")

    # Decide top of shading: a bit above max SWE
    if ylim_ is not None:
        y_top = ylim_
    else:
        swe_max = np.nanmax(swe_vals)
        margin = max(0.1, 0.2 * swe_max)  # at least 0.1 m, or 20%
        y_top = swe_max + margin
    if ylim_ is None:
        ax1.set_ylim(bottom=0.0, top=y_top)
    else:
        ax1.set_ylim(bottom=0.0, top=ylim_)

    # Helper: shade a phase from SWE to y_top where mask is True
    def _shade_phase(mask_da, color, alpha):
        m = mask_da.values.astype(bool)
        if m.size == 0:
            return
        # Also require SWE to be finite
        valid = m & np.isfinite(swe_vals)
        if not np.any(valid):
            return
        ax1.fill_between(
            t_vals,
            swe_vals,
            y_top,
            where=valid,
            color=color,
            alpha=alpha,
            interpolate=True,
            edgecolor="none",  # Remove edge lines
        )

    # Shade in priority order (last drawn is highest priority)
    _shade_phase(is_moist, color="lightblue", alpha=0.4)
    _shade_phase(is_ripening, color="palegoldenrod", alpha=0.5)
    _shade_phase(is_output, color="lightcoral", alpha=0.4)

    # Labels, grid, title
    ax1.set_ylabel("SWE (m)")
    if title is not None:
        ax1.set_title(title)
    ax1.grid(alpha=0.3)
    if fig is not None:
        fig.autofmt_xdate()

    # Build legend
    phase_patches = [
        Patch(facecolor="lightcoral", alpha=0.4, label="Output"),
        Patch(facecolor="palegoldenrod", alpha=0.5, label="Ripening"),
        Patch(facecolor="lightblue", alpha=0.4, label="Moistening"),
    ]

    handles = phase_patches
    labels = [p.get_label() for p in phase_patches]
    uniq = dict(zip(labels, handles))
    if ax is None:
        ax1.legend(uniq.values(), uniq.keys(), loc="lower right")

    if fig is not None:
        plt.tight_layout()
        plt.show()
        return
    else:
        return uniq, y_top


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

def wet_snow_by_orbit(path,cues_y,cues_x,orbit_,showOutput = False):
    # Constants to be used in the weights calculation
    k = 0.5
    theta1 = 20
    theta2 = 45

    binary_lst = []
    for file in sorted(os.listdir(path)):
        if showOutput: print(file)
        ## load and process sentinel ##
        sentinel_ds = xr.open_zarr(path + file)
        sentinel_ds = sentinel_ds.rio.write_crs("EPSG:32611")

        W = xr.where(sentinel_ds.local_incidence_angle<theta1,1,k*(1 + ((theta2-sentinel_ds.local_incidence_angle)/(theta2-theta1)) ) )
        W = xr.where(sentinel_ds.local_incidence_angle>theta2,k,W)
        Rc = W*sentinel_ds.ratio_images.sel(band='vh').groupby('sat:relative_orbit') + (1-W)*sentinel_ds.ratio_images.sel(band='vv').groupby('sat:relative_orbit') 
        Rc = Rc.rio.write_crs(sentinel_ds.rio.crs)
        binary_wet_snow = xr.where(Rc<-2,1,0)
        orbit_slice = binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True) \
                          .expand_dims({'Time':len(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time)}) \
                          .assign_coords({'Time':(binary_wet_snow.where(binary_wet_snow['sat:relative_orbit'] == orbit_,drop = True).time.values)})
        binary_lst.append(orbit_slice)

    wet_snow_tseries = xr.concat(binary_lst,dim = 'Time').drop_duplicates(dim = 'Time')
    
    idy_mammoth,idx_mammoth = sm_intersect_rectilinear(wet_snow_tseries.y.values,wet_snow_tseries.x.values,cues_y,cues_x)

    da_tseries = wet_snow_tseries.isel({'y':idy_mammoth,'x':idx_mammoth})[:,0]
    da_tseries.load()

    # melt_time = da_tseries.where(da_tseries == 1,drop = True)[0].Time.values
    melt_time = da_tseries.where(da_tseries == 1,drop = True).Time.values

    return wet_snow_tseries,da_tseries,melt_time

def snowmodel_output_phase(modeled_swe_corrected,modeled_runoff_corrected):
    swe  = modeled_swe_corrected.groupby(modeled_swe_corrected.time.dt.date).mean()
    roff  = modeled_runoff_corrected.groupby(modeled_runoff_corrected.time.dt.date).mean()['roff']

    # Align just in case
    swe, roff = xr.align(swe, roff)

    # 1) Peak SWE time
    t_peak = swe.idxmax('date')

    # 2) Differences
    swe_diff  = swe.diff('date')
    roff_diff = roff.diff('date')

    # 3) Only after the peak
    swe_diff_after  = swe_diff.sel(date=slice(t_peak, None))
    roff_diff_after = roff_diff.sel(date=slice(t_peak, None))

    # 4) Condition: SWE decreasing, runoff increasing
    eps_swe  = 0.0   # or small threshold
    eps_roff = 0.0
    cond = (swe_diff_after < -eps_swe) & (roff_diff_after > eps_roff)

    # 5) Check if there is any True at all
    if bool(cond.any().compute()):
        # index of first True
        onset_idx = int(cond.argmax('date'))
        melt_onset_time = cond['date'].isel(date=onset_idx).item()
    else:
        melt_onset_time = None

    print("SWE peak at:", t_peak.values)
    print("First time SWE↓ & runoff↑:", np.datetime64(melt_onset_time,'ns'))
    return t_peak.values,np.datetime64(melt_onset_time,'ns')

def snowmodel_ripening_phase(modeled_liq_corrected,sm_nc_corrected_dir,start_date,end_date,showOutput = False):

    temp_all = xr.load_dataset(sm_nc_corrected_dir + 'multilayer_temp.nc')
    temp_all = temp_all.where(temp_all.time >= start_date,drop = True).where(temp_all.time <= end_date,drop = True) \
                          .isel({'south_north':0,'east_west':0})
    temp_all = temp_all.where(temp_all['tsfc'] > -9998.0)['tsfc']
    
    mask_t = ~temp_all.isnull().values

    modeled_liq_corrected = modeled_liq_corrected.melt.where(mask_t)
    
    liq = modeled_liq_corrected  # 'melt' (time, layer)

    tol = 0.0  # or a small threshold like 1e-8 if needed

    # 1) Boolean mask of valid (non-NaN) liquid values
    valid = liq.notnull()                            # (time, layer)

    # 2) Layer indices as a DataArray, broadcast over time
    layer_idx = liq['layer']                         # (layer,)
    # xarray will broadcast this automatically when combined with (time, layer) arrays

    # 3) For each time, find the *deepest* non-NaN layer index
    #    valid*layer_idx -> layer number where valid, 0 where not
    last_valid_layer = (valid * layer_idx).where(valid).max(dim="layer")
    # last_valid_layer: (time,) gives the index of the deepest non-NaN layer
    # (0 if there are no valid layers at that time)

    # 4) Mask for liquid water above the deepest non-NaN layer
    inner_liq = (liq > tol) & (layer_idx < last_valid_layer)

    # 5) For each time, is there ANY inner layer with liquid?
    has_inner_liq = inner_liq.any(dim="layer")

    # If dask-backed, compute this 1D mask so we can safely index with it
    has_inner_liq = has_inner_liq.compute()

    # 6) Times (and indices) where condition is met
    times_with_inner_liq = liq['time'].where(has_inner_liq, drop=True)
    time_indices = np.where(has_inner_liq.values)[0]

    times_utc = modeled_liq_corrected['time'].isel(time=time_indices).to_index()

    times_local = (
        times_utc
        .tz_localize("UTC")                 # mark as UTC
        .tz_convert("America/Los_Angeles")  # convert to Pacific time
    )
    times_local_naive = times_local.tz_localize(None)

    if showOutput:
        print("Times with liquid in an internal (non-last-valid) layer:")
        print(times_with_inner_liq.values)
        print("Times local")
        print(times_local_naive.values)

        print("Corresponding time indices:")
        print(time_indices)

    ripe_windows = snowmodel_ripening_phase_end(liq,has_inner_liq)
    return ripe_windows

def snowmodel_ripening_phase_end(liq,has_inner_liq):
    time_index = liq['time'].to_index()

    # -------------------------------------------------------------------
    # 1) Collapse to daily: is there inner liquid at ANY timestep that day?
    # -------------------------------------------------------------------
    daily_inner = has_inner_liq.resample(time="1D").any()  # (time=days)
    s = daily_inner.to_series().astype(bool)               # pandas Series

    # -------------------------------------------------------------------
    # 2) Find runs of consecutive True days
    #    (segment ID changes whenever True/False flips)
    # -------------------------------------------------------------------
    groups = (s != s.shift()).cumsum()

    windows = []  # will hold (start_time, end_time, duration_hours)

    for gid, seg in s.groupby(groups):
        if not seg.iloc[0]:
            # this segment is all False days – skip
            continue

        # seg is a run of consecutive True days
        day_start = seg.index[0]
        day_end   = seg.index[-1]

        # ----------------------------------------------------------------
        # 3) Within these days, find the first and last *3-hr* timesteps
        #    with inner liquid
        # ----------------------------------------------------------------
        # Mask on the original 3-hr time axis
        in_days = (time_index >= day_start) & (time_index < day_end + pd.Timedelta(days=1))
        mask = in_days & has_inner_liq.values

        if not mask.any():
            continue  # shouldn’t happen, but just in case

        start_time = time_index[mask].min()
        end_time   = time_index[mask].max()

        duration = end_time - start_time
        duration_hours = duration / np.timedelta64(1, "h")

        # ----------------------------------------------------------------
        # 4) Keep only windows longer than 24 hours
        # ----------------------------------------------------------------
        if duration_hours > 24.0:
            windows.append((start_time, end_time, float(duration_hours)))

    # windows now contains your sequences
    for i, (t0, t1, dur) in enumerate(windows, 1):
        pass
        # print(f"Window {i}: {t0} → {t1}  (duration ~ {dur:.1f} h)")
    return windows

def plot_snowmelt_phases(
    ds,
    ds_phase,
    swe_var="swed",
    runoff_var="roff",
    title=None,
    t_start=None,
    t_end=None,
    figsize=(14, 4),
):
    """
    Plot SWE and runoff with shaded snowmelt phases.

    Parameters
    ----------
    ds : xarray.Dataset
        Original dataset with SWE and runoff.
    ds_phase : xarray.Dataset
        Output from classify_snowmelt_phases_3h_full, must contain phase_code_3h.
    swe_var : str
        Name of SWE variable in ds (default 'swed').
    runoff_var : str
        Name of runoff variable in ds (default 'roff').
    title : str or None
        Figure title.
    t_start, t_end : str or np.datetime64 or None
        Optional time bounds for zooming (e.g., '2017-04-01', '2017-05-01').
    figsize : tuple
        Figure size.
    """

    # --- time selection / zoom ---
    if t_start is not None or t_end is not None:
        tsel = slice(t_start, t_end)
        ds_ = ds.sel(time=tsel)
        ph_ = ds_phase.sel(time=tsel)
    else:
        ds_ = ds
        ph_ = ds_phase

    time   = ds_["time"]
    swe    = ds_[swe_var]
    runoff = ds_[runoff_var]

    phase_code = ph_["phase_code_3h"]

    # Boolean masks by phase code
    is_output    = phase_code == 3
    is_ripening  = phase_code == 2
    is_moist     = phase_code == 1
    # dry would be phase_code == 0, but we don't shade it

    # --- plotting ---
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # SWE and runoff
    ax1.plot(time, swe, "k-", linewidth=1.6, label="SWE (m)")
    ax2.plot(time, runoff, "b--", alpha=0.5, label="Runoff")

    # Helper: shade contiguous True segments
    def _shade_mask(mask, color, alpha):
        s = mask.to_series().astype(bool)
        if s.empty:
            return
        groups = (s != s.shift()).cumsum()
        for _, seg in s.groupby(groups):
            if seg.iloc[0]:
                ax1.axvspan(seg.index[0], seg.index[-1],
                            color=color, alpha=alpha)

    # Shade phases (order from bottom to top visually if they overlap)
    _shade_mask(is_moist,    color="lightblue",     alpha=0.4)
    _shade_mask(is_ripening, color="palegoldenrod", alpha=0.5)
    _shade_mask(is_output,   color="lightcoral",    alpha=0.4)

    # Labels, grid, title
    ax1.set_ylabel("SWE (m)")
    ax2.set_ylabel("Runoff")
    if title is not None:
        ax1.set_title(title)
    ax1.grid(alpha=0.3)
    fig.autofmt_xdate()

    # Build legend: lines + phase patches
    line_handles, line_labels = ax1.get_legend_handles_labels()
    line_handles2, line_labels2 = ax2.get_legend_handles_labels()

    phase_patches = [
        Patch(facecolor="lightcoral",    alpha=0.4, label="Output"),
        Patch(facecolor="palegoldenrod", alpha=0.5, label="Ripening"),
        Patch(facecolor="lightblue",     alpha=0.4, label="Moistening"),
    ]

    handles = line_handles + line_handles2 + phase_patches
    labels  = line_labels  + line_labels2  + [p.get_label() for p in phase_patches]

    # remove duplicate labels
    uniq = dict(zip(labels, handles))
    ax1.legend(uniq.values(), uniq.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()
    return

def model_sar_obs_cues_timeseries_morning_afternoon(sm_cues_corrected_fpath: str,
                                  cues_obs_swe_df: pd.DataFrame,
                                  df_lys: pd.DataFrame,
                                  sentinel_cues: xr.DataArray,
                                  start_date: np.ndarray,
                                  end_date: np.ndarray,
                                  idy_cues: int,
                                  idx_cues: int,
                                  cues_y: int,
                                  cues_x: int,
                                  idy_: int = 0,
                                  idx_: int = 0,
                                  showLayers: bool = False,
                                  vmin_: int = -15,
                                  vmax_:int = 0,
                                  cmap_ = 'inferno',
                                  is3x3Mean: bool = False,
                                  show_S1_thresh: bool = False,
                                  S1_thresh_path: str = None,
                                  ):
    """
    time series of data at cues.
    Input:
      polarization - string of polarization ("vv","vh")
      sentinel_ds - sentinel xarray dataset.
      month_start - integer for start month.
      month_end - integer for end month.
      geom_proj - subdomain geometry.
      point_idx - point x-coordinate.
      point_idx - point y-coordinate.
      domain - subdomain string.
    Output:
      std_arr - numpy array for mean standard deviation.
    """
    wy = int(str(end_date)[0:4])
    
    color_dict = {64:'C0',
                 137:'C1',
                 144:'C2'
                 }
    
    swe_ymax = (cues_obs_swe_df[(cues_obs_swe_df['DateTime'] >= start_date) & (cues_obs_swe_df['DateTime'] <= end_date)]['SWE'].max() / 1000) * 1.4
    
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

    sm_nc_dir = f'{sm_cues_corrected_fpath}{snowmodel_dict[wy]}/netcdf/'

    temp_ml_ds, swed_top, swed_ml_ds = visualize_3D_sm_var('multilayer_swed',
                            'multilayer_temp',
                            sm_nc_dir,
                            0,
                            0,
                           start_date = start_date,
                           end_date = end_date,
                           showPlot = False,
                           plotLWC = False,
                           )
    sm_ml_liq = xr.load_dataset(f'{sm_nc_dir}/multilayer_liq.nc')
    sm_sl_smlt = xr.load_dataset(f'{sm_nc_dir}/smlt.nc')
    sm_sl_roff = xr.load_dataset(f'{sm_nc_dir}/roff.nc')
    
    start_day_wr = np.datetime64(f'{int(str(end_date)[0:4])-1}-10-01')
    end_day_wr = np.datetime64(f'{int(str(end_date)[0:4])}-10-01')

    modeled_swe_corrected = swed_top.where(swed_top.time >= start_day_wr,drop = True).where(swed_top.time <= end_day_wr,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_}).sum(dim = 'layer')
    modeled_runoff_corrected = sm_sl_roff.where(sm_sl_roff.time >= start_day_wr,drop = True).where(sm_sl_roff.time <= end_day_wr,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    modeled_temp_corrected = temp_ml_ds.where(temp_ml_ds.time >= start_day_wr,drop = True).where(temp_ml_ds.time <= end_day_wr,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    modeled_liq_corrected = sm_ml_liq.where(sm_ml_liq.time >= start_day_wr,drop = True).where(sm_ml_liq.time <= end_day_wr,drop = True) \
                          .isel({'south_north':idy_,'east_west':idx_})
    
    # # identify snowmodel output dates
    # swemax_t, output_init_t = snowmodel_output_phase(modeled_swe_corrected,modeled_runoff_corrected)
    
    # swe_melt = modeled_swe_corrected.where(modeled_swe_corrected.time >= output_init_t,drop = True)
    # swe_melt.load()
    # sm_melt_out_date = swe_melt.where(swe_melt <= 0.2,drop = True).time[0].values
    # print(f'SnowModel Output Phase',sm_melt_out_date)

    # # identify ripening windows.
    # ripe_windows = snowmodel_ripening_phase(modeled_liq_corrected,sm_nc_dir,start_day_wr,end_day_wr,showOutput = False)
    # print(f'SnowModel Ripening Windows',ripe_windows)
    
    fig = plt.figure(figsize=(12, 12))  # Taller figure to fit all plots

    gs = gridspec.GridSpec(
        5, 2,  # 4 rows, 2 columns
        width_ratios=[20, 1],
        height_ratios=[3, 1, 1, 1, 1],  # main plot taller
        wspace=0.05,
        hspace=0.3
    )

    # Main axes (top)
    ax0 = fig.add_subplot(gs[0, 0])

    # Additional axes below
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    # Colorbar axes next to the top plot
    cax = fig.add_subplot(gs[0, 1])
    pos = cax.get_position()  # Get [left, bottom, width, height]
    cax.set_position([pos.x0 + 0.02, pos.y0, pos.width * 0.5, pos.height])

    # 4️⃣ Main time series
    l1, = swed_top[:,:,idy_,idx_].sum(dim='layer').plot(ax=ax0, label='SWE', color='black')

    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()

    # 5️⃣ Top layer gradient fill
    gradient = fill(
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].time.values,
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].values,
        swed_top[:,:,idy_,idx_].sum(dim='layer').values,
        temp_ml_ds[:,-1,idy_,idx_].values,
        vmin_,
        vmax_,
        axes=ax0,
        cmap_=cmap_
    )

# Remaining layers
    for i in range(1, swed_ml_ds.shape[1]):
        gradient = fill(
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].time.values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i-1].values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].values,
            temp_ml_ds[:,-i-1,idy_,idx_].values,
            vmin_,
            vmax_,
            axes=ax0,
            cmap_=cmap_
    )

    # Colorbar
    cbar = fig.colorbar(gradient, cax=cax, fraction=0.01, pad=0.05)
    cbar.set_label("Modeled Temperature (°C)")

    # Observations
    l4, = ax0.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE')
    # ax0.fill_between([output_init_t,sm_melt_out_date], [swe_ymax,swe_ymax], 0, color='lightgray', alpha=0.5, label='Output Phase')
    # ax0.fill_between(ripe_windows[0][0:2], [swe_ymax,swe_ymax], 0, color='blue', alpha=0.5, label='Output Phase')
    # ax0.axvline(output_init_t,linestyle = '--',color = 'black')
    l2, = sm_sl_roff.where(sm_sl_roff.time >= start_date,drop = True).where(sm_sl_roff.time <= end_date,drop = True).roff[:,0,0].cumsum().plot(ax=ax0,color = 'C2',label = 'Runoff')
    # ax0.legend()
    # Axes formatting
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_ylim(0,swe_ymax)
    ax0.grid(True, linestyle='--')
    if (start_date is not None) and (end_date is not None):
        ax0.set_xlim([start_date, end_date])
    ax0.set_title("Modeled and Observed SWE",fontweight = 'bold')
    ax0.set_ylabel("SWE\n[m]",fontweight = 'bold')
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_xlabel("")

    # liquid water in snow column second axis.
    ax0_ = ax0.twinx()
    l3, = sm_ml_liq.where(sm_ml_liq.time >= start_date,drop = True).where(sm_ml_liq.time <= end_date,drop = True).where(sm_ml_liq.sum(dim = 'layer').melt[:,0,0] > 0).sum(dim='layer').melt[:,0,0].plot(ax=ax0_)
    ax0_.set_ylabel('Total Liquid Water in Column\n[m]',fontweight = 'bold',color = 'C0')
    ax0_.set_title('')

    ax0_.set_yticks([0.005, 0.010, 0.015])
    ax0_.tick_params(axis='y', direction='in', pad=-40)
    ax0_.set_ylim(0.0,0.02)
    leg1 = ax0.legend(handles = [l1,l2,l3],title = 'Modeled',loc = 'upper left',title_fontproperties={'weight': 'bold'})
    ax0.add_artist(leg1)
    leg2 = ax0.legend(handles=[l4], title='Observed',loc = 'upper right',title_fontproperties={'weight': 'bold'})

    # 6️⃣ Example content for ax1
    # Replace this with your actual data plotting in ax1
    # Here is an example bar plot with a legend
    # df_lys = df_lys.reset_index()
    df_lys_ = df_lys[(df_lys['datetime'] >= start_date) & (df_lys['datetime'] <= end_date)]
    dates = df_lys_['datetime']
    # ax1.bar(dates, np.random.randint(0, 100, size=len(dates)), width=1, label="Example Series")

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb1_south", "tb2_south", "tb3_south", "tb4_south"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax1.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax1.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax1.grid(True, linestyle="--")
    ax1.set_title("Observed Southern Lysimeters",fontweight = 'bold')
    ax1.tick_params(axis='x', labelbottom=False)

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb5_north", "tb6_north", "tb7_north", "tb8_north"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax2.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax2.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax2.grid(True, linestyle="--")
    ax2.set_title("Observed Northern Lysimeters",fontweight = 'bold')
    ax2.tick_params(axis='x', labelbottom=False)

    sentinel_cues_yr = sentinel_cues.where(sentinel_cues.time >= start_date,drop = True)
    sentinel_cues_yr = sentinel_cues_yr.where(sentinel_cues_yr.time <= end_date,drop = True)
    # tmin1 = sentinel_cues_yr.time[-1].values
    # tmax1 = sentinel_cues_yr.time[0].values
    ## morning ##
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date1 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        
        if is3x3Mean:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idx_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
            

        ax3.plot(date1,vals1,label = f'{orbit}, vv',color = color_dict[orbit],marker = '*')

        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date1)):
                if date1[__date] in melt_time_orbit:
                    # pass
                    ax3.scatter(date1[__date],vals1[__date],label = None,color = 'red',marker = '*',s = 100)
            # date_msk1 = (date1 == melt_time_orbit)
            # date_msk1 = (date1.isin(melt_time_orbit))
            # ax3.scatter(date1[date_msk1],vals1[date_msk1],label = None,color = 'red',marker = '*',s = 100)
        



    # tmin2 = sentinel_cues_yr.time[-1].values
    # tmax2 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date2 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        if is3x3Mean:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idy_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
        ax3.plot(date2,vals2,label = f'{orbit}, vh',color = color_dict[orbit],linestyle = '--',marker = '*')
    
        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date2)):
                if date2[__date] in melt_time_orbit:
                    # pass
                    ax3.scatter(date2[__date],vals2[__date],label = None,color = 'red',marker = '*',s = 100)
            # date_msk2 = (date2 == melt_time_orbit)
            # date_msk2 = (date2.isin(melt_time_orbit))
            # ax3.scatter(date2[date_msk2],vals2[date_msk2],label = None,color = 'red',marker = '*',s = 100)

    ax3.set_ylabel("Backscatter\n[dB]",fontweight = 'bold')
    ax3.set_title("S1 - Morning VV/VH",fontweight = 'bold')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.grid(True, linestyle="--")
    ax3_ = ax3.twinx()
    ax3_.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE')
    ax3_.set_ylabel('SWE [m]',fontweight = 'bold',color = 'red')

    ## afternoon ##
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date1 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
    
        if is3x3Mean:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idx_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
    
        ax4.plot(date1,vals1,label = f'{orbit}, vv',color = color_dict[orbit],marker = '*')
        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date1)):
                if date1[__date] in melt_time_orbit:
                    # pass
                    ax4.scatter(date1[__date],vals1[__date],label = None,color = 'red',marker = '*',s = 100)
            # date_msk1 = (date1 == melt_time_orbit)
            # date_msk1 = (date1.isin(melt_time_orbit))
            # ax4.scatter(date1[date_msk1],vals1[date_msk1],label = None,color = 'red',marker = '*',s = 100)



    # tmin2 = sentinel_cues_yr.time[-1].values
    # tmax2 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date2 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        if is3x3Mean:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idy_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
        ax4.plot(date2,vals2,label = f'{orbit}, vh',color = color_dict[orbit],linestyle = '--',marker = '*')

        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date2)):
                if date2[__date] in melt_time_orbit:
                    # pass
                    ax4.scatter(date2[__date],vals2[__date],label = None,color = 'red',marker = '*',s = 100)
            # date_msk2 = (date2 == melt_time_orbit)
            # date_msk2 = (date2.isin(melt_time_orbit))
            # ax4.scatter(date2[date_msk2],vals2[date_msk2],label = None,color = 'red',marker = '*',s = 100)
    ax4.set_ylabel("Backscatter\n[dB]",fontweight = 'bold')
    ax4.set_title("S1 - Afternoon VV/VH",fontweight = 'bold')
    ax4.tick_params(axis='x', labelbottom=True)
    ax4.grid(True, linestyle="--")
    ax4_ = ax4.twinx()
    ax4_.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE')
    ax4_.set_ylabel('SWE [m]',fontweight = 'bold',color = 'red')

    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax1.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax2.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    ax3.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    # ax4.legend(
    #     loc='upper left',
    #     bbox_to_anchor=(1.02, 1.0),
    #     borderaxespad=0
    # )


    plt.tight_layout()
    plt.show()  
    
    return 

def orbit_state_mapping(ds_ref:xr.Dataset):
    # Convert the two coords to a tidy dataframe
    df = (
        ds_ref[["sat:relative_orbit", "sat:orbit_state"]]
        .to_dataframe()
        .reset_index()[["sat:relative_orbit", "sat:orbit_state"]]
    )

    # Keep only unique (orbit, state) pairs
    df_unique = df.drop_duplicates()

    # (Optional) check that each orbit only has one state
    check = df_unique.groupby("sat:relative_orbit")["sat:orbit_state"].nunique()
    if (check > 1).any():
        raise ValueError("Some sat:relative_orbit values map to multiple orbit states!")

    # Build the mapping dict: {144: 'descending', 137: 'ascending', 64: 'ascending', ...}
    orbit_mapping = (
        df_unique.set_index("sat:relative_orbit")["sat:orbit_state"]
        .to_dict()
    )

    # Optionally ensure integer keys
    orbit_mapping = {int(k): v for k, v in orbit_mapping.items()}
    return orbit_mapping


def map_orbit_state_reference_image(s1_tseries: xr.Dataset,
                                    s1_reference: xr.Dataset):
    
    
    mapping = orbit_state_mapping(s1_tseries)

    # Create the new coordinate indexed by sat:relative_orbit dimension
    orbit_state = s1_reference["sat:relative_orbit"].to_series().map(mapping).to_xarray()

    # Assign it as a new coordinate
    return s1_reference.assign_coords({"sat:orbit_state": orbit_state})

def model_sar_obs_cues_timeseries_morning_afternoon_shade(sm_cues_corrected_fpath: str,
                                  cues_obs_swe_df: pd.DataFrame,
                                  df_lys: pd.DataFrame,
                                  sentinel_cues: xr.DataArray,
                                  snowmodel_cues: xr.DataArray,
                                  snowmodel_phases: xr.DataArray,
                                  s1_melt: xr.DataArray,
                                  s1_phases: xr.DataArray,
                                  start_date: np.ndarray,
                                  end_date: np.ndarray,
                                  idy_cues: int,
                                  idx_cues: int,
                                  cues_y: int,
                                  cues_x: int,
                                  idy_: int = 0,
                                  idx_: int = 0,
                                  showLayers: bool = False,
                                  vmin_: int = -15,
                                  vmax_:int = 0,
                                  cmap_ = 'inferno',
                                  is3x3Mean: bool = False,
                                  show_S1_thresh: bool = False,
                                  S1_thresh_path: str = None,
                                  S1_ref_ds: xr.Dataset = None,
                                  ):
    """
    time series of data at cues.
    Input:
      polarization - string of polarization ("vv","vh")
      sentinel_ds - sentinel xarray dataset.
      month_start - integer for start month.
      month_end - integer for end month.
      geom_proj - subdomain geometry.
      point_idx - point x-coordinate.
      point_idx - point y-coordinate.
      domain - subdomain string.
    Output:
      std_arr - numpy array for mean standard deviation.
    """
    wy = int(str(end_date)[0:4])
    
    color_dict = {64:'C0',
                 137:'C1',
                 144:'C2'
                 }
    
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

    sm_nc_dir = f'{sm_cues_corrected_fpath}{snowmodel_dict[wy]}/netcdf/'
    
    swe_ymax = (cues_obs_swe_df[(cues_obs_swe_df['DateTime'] >= start_date) & (cues_obs_swe_df['DateTime'] <= end_date)]['SWE'].max() / 1000) * 1.4

    temp_ml_ds, swed_top, swed_ml_ds = visualize_3D_sm_var('multilayer_swed',
                            'multilayer_temp',
                            sm_nc_dir,
                            0,
                            0,
                           start_date = start_date,
                           end_date = end_date,
                           showPlot = False,
                           plotLWC = False,
                           )
    
    # map sat:orbit_state to reference image.
    if S1_ref_ds is not None:
        S1_ref_ds = map_orbit_state_reference_image(sentinel_cues,S1_ref_ds)
    
    
    fig = plt.figure(figsize=(12, 12))  # Taller figure to fit all plots

    gs = gridspec.GridSpec(
        5, 2,  # 4 rows, 2 columns
        width_ratios=[20, 1],
        height_ratios=[3, 1, 1, 1, 1],  # main plot taller
        wspace=0.05,
        hspace=0.3
    )

    # Main axes (top)
    ax0 = fig.add_subplot(gs[0, 0])

    # Additional axes below
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    # Colorbar axes next to the top plot
    cax = fig.add_subplot(gs[0, 1])
    pos = cax.get_position()  # Get [left, bottom, width, height]
    cax.set_position([pos.x0 + 0.02, pos.y0, pos.width * 0.5, pos.height])

    # 4️⃣ Main time series
    l1, = snowmodel_cues.swed.plot(ax=ax0, label='SWE', color='black')

    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()

    # 5️⃣ Top layer gradient fill
    gradient = fill(
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].time.values,
        swed_ml_ds.where(swed_ml_ds >= 0)[:,0].values,
        swed_top[:,:,idy_,idx_].sum(dim='layer').values,
        temp_ml_ds[:,-1,idy_,idx_].values,
        vmin_,
        vmax_,
        axes=ax0,
        cmap_=cmap_
    )

# Remaining layers
    for i in range(1, swed_ml_ds.shape[1]):
        gradient = fill(
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].time.values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i-1].values,
            swed_ml_ds.where(swed_ml_ds >= 0)[:,i].values,
            temp_ml_ds[:,-i-1,idy_,idx_].values,
            vmin_,
            vmax_,
            axes=ax0,
            cmap_=cmap_
    )

    # Colorbar
    cbar = fig.colorbar(gradient, cax=cax, fraction=0.01, pad=0.05)
    cbar.set_label("Modeled Temperature (°C)")

    # Observations
    l4, = ax0.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE')
    # ax0.fill_between([output_init_t,sm_melt_out_date], [swe_ymax,swe_ymax], 0, color='lightgray', alpha=0.5, label='Output Phase')
    # ax0.fill_between(ripe_windows[0][0:2], [swe_ymax,swe_ymax], 0, color='blue', alpha=0.5, label='Output Phase')
    # ax0.axvline(output_init_t,linestyle = '--',color = 'black')
    l2, = snowmodel_cues.where(snowmodel_cues.time >= start_date,drop = True).where(snowmodel_cues.time <= end_date,drop = True).roff.cumsum().plot(ax=ax0,color = 'C2',label = 'Runoff')
    # ax0.legend()
    # Axes formatting
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)

    ax0.grid(True, linestyle='--')
    if (start_date is not None) and (end_date is not None):
        ax0.set_xlim([start_date, end_date])
    ax0.set_title("Modeled and Observed SWE",fontweight = 'bold')
    ax0.set_ylabel("SWE\n[m]",fontweight = 'bold')
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_xlabel("")

    # plot snowmodel phases.
    uniq,swe_ymax = plot_snowmelt_phases_filltop(snowmodel_cues, snowmodel_phases, ax=ax0)
    # liquid water in snow column second axis.
    ax0_ = ax0.twinx()
    l3, = snowmodel_cues.where(snowmodel_cues.time >= start_date,drop = True).where(snowmodel_cues.time <= end_date,drop = True).where(snowmodel_cues.sum(dim = 'layer').melt > 0).sum(dim='layer').melt.plot(ax=ax0_)
    ax0_.set_ylabel('Total Liquid Water in Column\n[m]',fontweight = 'bold',color = 'C0')
    ax0_.set_title('')

    ax0_.set_yticks([0.005, 0.010, 0.015])
    ax0_.tick_params(axis='y', direction='in', pad=-40)
    ax0_.set_ylim(0.0,0.02)
    leg1 = ax0.legend(handles = [l1,l2,l3],title = 'Modeled',loc = 'upper left',title_fontproperties={'weight': 'bold'})
    ax0.add_artist(leg1)
    leg2 = ax0.legend(handles=[l4], title='Observed',loc = 'upper right',title_fontproperties={'weight': 'bold'})
    ax0.add_artist(leg2)
    leg3 = ax0.legend(uniq.values(), uniq.keys(), loc="lower right",title = 'Melt Phases',title_fontproperties={'weight': 'bold'})
    ax0.set_ylim(0,swe_ymax)


    # 6️⃣ Example content for ax1
    # Replace this with your actual data plotting in ax1
    # Here is an example bar plot with a legend
    # df_lys = df_lys.reset_index()
    df_lys_ = df_lys[(df_lys['datetime'] >= start_date) & (df_lys['datetime'] <= end_date)]
    dates = df_lys_['datetime']
    # ax1.bar(dates, np.random.randint(0, 100, size=len(dates)), width=1, label="Example Series")

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb1_south", "tb2_south", "tb3_south", "tb4_south"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax1.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax1.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax1.grid(True, linestyle="--")
    ax1.set_title("Observed Southern Lysimeters",fontweight = 'bold')
    ax1.tick_params(axis='x', labelbottom=False)

    # Add multiple series to illustrate legend
    for label, color in zip(
    ["tb5_north", "tb6_north", "tb7_north", "tb8_north"],
    ["blue", "orange", "green", "red", "purple"]
    ):
        ax2.plot(dates, df_lys_.replace(0, np.nan)[label], label=label, color=color)

    ax2.set_ylabel("Melt water eq.\n[cm]",fontweight = 'bold')
    ax2.grid(True, linestyle="--")
    ax2.set_title("Observed Northern Lysimeters",fontweight = 'bold')
    ax2.tick_params(axis='x', labelbottom=False)

    sentinel_cues_yr = sentinel_cues.where(sentinel_cues.time >= start_date,drop = True)
    sentinel_cues_yr = sentinel_cues_yr.where(sentinel_cues_yr.time <= end_date,drop = True)
    # tmin1 = sentinel_cues_yr.time[-1].values
    # tmax1 = sentinel_cues_yr.time[0].values
    
    ## morning ##
    # ax3_ = ax3.twinx()
    # ax3_.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE',alpha = 0.5)
    # ax3_.set_ylabel('SWE [m]',fontweight = 'bold',color = 'red')
    # ax3_.set_ylim(0,swe_ymax)
    _,__ = plot_sentinel_melt_phases_continuous(s1_melt,s1_phases,ax=ax3)
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date1 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        
        if is3x3Mean:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idx_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == 'descending',drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
            
        # plot reference line.
        if S1_ref_ds is not None:
            reference_bsc = (10*np.log10(S1_ref_ds.where(S1_ref_ds["sat:relative_orbit"] == orbit,drop = True) \
                                     .where(S1_ref_ds["sat:orbit_state"] == 'descending',drop = True) \
                                     .where(S1_ref_ds["band"] == 'vv',drop = True) \
                                     .isel({'y':idy_cues,'x':idx_cues})['reference_bscatter'].values))
            if len(reference_bsc) > 0:
                ax3.axhline(reference_bsc[0],color = color_dict[orbit],linestyle = '--',linewidth = 0.75)
            
        # plot backscatter.
        ax3.plot(date1,vals1,label = f'{orbit}, vv',color = color_dict[orbit],marker = '*')

        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date1)):
                if date1[__date] in melt_time_orbit:
                    # pass
                    ax3.scatter(date1[__date],vals1[__date],label = None,color = 'purple',marker = '*',s = 100)
            # date_msk1 = (date1 == melt_time_orbit)
            # date_msk1 = (date1.isin(melt_time_orbit))
            # ax3.scatter(date1[date_msk1],vals1[date_msk1],label = None,color = 'red',marker = '*',s = 100)
        



    # tmin2 = sentinel_cues_yr.time[-1].values
    # tmax2 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date2 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        if is3x3Mean:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idy_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "descending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
        # plot reference line.
        if S1_ref_ds is not None:
            reference_bsc = (10*np.log10(S1_ref_ds.where(S1_ref_ds["sat:relative_orbit"] == orbit,drop = True) \
                                     .where(S1_ref_ds["sat:orbit_state"] == 'descending',drop = True) \
                                     .where(S1_ref_ds["band"] == 'vh',drop = True) \
                                     .isel({'y':idy_cues,'x':idx_cues})['reference_bscatter'].values))
            if len(reference_bsc) > 0:
                ax3.axhline(reference_bsc[0],color = color_dict[orbit],linestyle = '--',linewidth = 0.75)
        
        # plot backscatter.
        ax3.plot(date2,vals2,label = f'{orbit}, vh',color = color_dict[orbit],linestyle = '--',marker = '*')
    
        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date2)):
                if date2[__date] in melt_time_orbit:
                    # pass
                    ax3.scatter(date2[__date],vals2[__date],label = None,color = 'purple',marker = '*',s = 100)
            # date_msk2 = (date2 == melt_time_orbit)
            # date_msk2 = (date2.isin(melt_time_orbit))
            # ax3.scatter(date2[date_msk2],vals2[date_msk2],label = None,color = 'red',marker = '*',s = 100)

    ax3.set_ylabel("Backscatter\n[dB]",fontweight = 'bold')
    ax3.set_title("S1 - Morning VV/VH",fontweight = 'bold')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.grid(True,axis = 'x', linestyle="--")
    

    ## afternoon ##
    # plot SWE on right yaxis
    # ax4_ = ax4.twinx()
    # ax4_.plot(cues_obs_swe_df['DateTime'], cues_obs_swe_df['SWE'] / 1000, color='red',label = 'SWE',alpha = 0.5)
    # ax4_.set_ylabel('SWE [m]',fontweight = 'bold',color = 'red')
    # ax4_.set_ylim(0,swe_ymax)

    _,__ = plot_sentinel_melt_phases_continuous(s1_melt,s1_phases,ax=ax4)
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date1 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
    
        if is3x3Mean:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idx_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals1 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vv',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
    
        # plot reference line.
        if S1_ref_ds is not None:
            reference_bsc = (10*np.log10(S1_ref_ds.where(S1_ref_ds["sat:relative_orbit"] == orbit,drop = True) \
                                     .where(S1_ref_ds["sat:orbit_state"] == 'ascending',drop = True) \
                                     .where(S1_ref_ds["band"] == 'vv',drop = True) \
                                     .isel({'y':idy_cues,'x':idx_cues})['reference_bscatter'].values))
            if len(reference_bsc) > 0:
                ax4.axhline(reference_bsc[0],color = color_dict[orbit],linestyle = '--',linewidth = 0.75)
        
        # plot backscatter.
        ax4.plot(date1,vals1,label = f'{orbit}, vv',color = color_dict[orbit],marker = '*')
        
        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date1)):
                if date1[__date] in melt_time_orbit:
                    # pass
                    ax4.scatter(date1[__date],vals1[__date],label = None,color = 'purple',marker = '*',s = 100)
            # date_msk1 = (date1 == melt_time_orbit)
            # date_msk1 = (date1.isin(melt_time_orbit))
            # ax4.scatter(date1[date_msk1],vals1[date_msk1],label = None,color = 'red',marker = '*',s = 100)



    # tmin2 = sentinel_cues_yr.time[-1].values
    # tmax2 = sentinel_cues_yr.time[0].values
    for orbit in np.unique(sentinel_cues_yr['sat:relative_orbit'].values):
        date2 = sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues].time.values
        if is3x3Mean:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,
                                                                                                             idy_cues-1:idy_cues+2,
                                                                                                             idx_cues-1:idy_cues+2].mean(dim = 'y').mean(dim = 'x'))) 
        else:
            vals2 = (10*np.log10(sentinel_cues_yr.where(sentinel_cues_yr["sat:orbit_state"] == "ascending",drop = True) \
                                    .where(sentinel_cues_yr["sat:relative_orbit"] == orbit,drop = True) \
                                            .where(sentinel_cues_yr["band"] == 'vh',drop = True)['bscatter'][:,0,idy_cues,idx_cues]))
        # plot reference line.
        if S1_ref_ds is not None:
            reference_bsc = (10*np.log10(S1_ref_ds.where(S1_ref_ds["sat:relative_orbit"] == orbit,drop = True) \
                                     .where(S1_ref_ds["sat:orbit_state"] == 'ascending',drop = True) \
                                     .where(S1_ref_ds["band"] == 'vh',drop = True) \
                                     .isel({'y':idy_cues,'x':idx_cues})['reference_bscatter'].values))
            if len(reference_bsc) > 0:
                ax4.axhline(reference_bsc[0],color = color_dict[orbit],linestyle = '--',linewidth = 0.75)
        # plot backscatter.
        ax4.plot(date2,vals2,label = f'{orbit}, vh',color = color_dict[orbit],linestyle = '--',marker = '*')

        if show_S1_thresh:
            wet_snow_orbit, tseries_orbit,melt_time_orbit = wet_snow_by_orbit(S1_thresh_path,cues_y,cues_x,orbit,showOutput = False)
            for __date in range(0,len(date2)):
                if date2[__date] in melt_time_orbit:
                    # pass
                    ax4.scatter(date2[__date],vals2[__date],label = None,color = 'purple',marker = '*',s = 100)
            # date_msk2 = (date2 == melt_time_orbit)
            # date_msk2 = (date2.isin(melt_time_orbit))
            # ax4.scatter(date2[date_msk2],vals2[date_msk2],label = None,color = 'red',marker = '*',s = 100)
    ax4.set_ylabel("Backscatter\n[dB]",fontweight = 'bold')
    ax4.set_title("S1 - Afternoon VV/VH",fontweight = 'bold')
    ax4.tick_params(axis='x', labelbottom=True)
    ax4.grid(True, axis = 'x', linestyle="--")


    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax1.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    # 7️⃣ Legend outside ax1 aligned with colorbar
    ax2.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    ax3.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    # ax4.legend(
    #     loc='upper left',
    #     bbox_to_anchor=(1.02, 1.0),
    #     borderaxespad=0
    # )


    plt.tight_layout()
    plt.show()  
    
    return


def fill(x, y1, y2, color_var, vmin_, vmax_, axes, cmap_='inferno'):
    if np.isnan(y2).any():
        mask = np.isnan(y2)
        y2[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            y2[~mask]
        )

    if np.isnan(color_var).any():
        mask = np.isnan(color_var)
        color_var[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            color_var[~mask]
        )

    polygon = axes.fill_between(x, y1, y2, lw=0, color='none')

    verts = np.vstack([p.vertices for p in polygon.get_paths()])

    gradient = axes.imshow(
        color_var.reshape(1, -1),
        cmap=cmap_,
        aspect='auto',
        vmin=vmin_,
        vmax=vmax_,
        extent=[
            verts[:, 0].min(),
            verts[:, 0].max(),
            verts[:, 1].min(),
            verts[:, 1].max()
        ]
    )

    gradient.set_clip_path(polygon.get_paths()[0], transform=axes.transData)

    return gradient

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

def plot_orbit_timeseries(ts_by_orbit, 
                          wet_thresh_db=-2.0, 
                          title="",
                          xmin_ = None,
                          xmax_ = None,
                          ):
    orbits = [o for o, ds in ts_by_orbit.items() if ds is not None]
    n = len(orbits)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2*n), dpi=160, sharex=True)
    # fig, axes = plt.subplots(n, 1, figsize=(6,3), dpi=200, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, o in zip(axes, orbits):
        ds = ts_by_orbit[o]
        t = pd.to_datetime(ds["Time"].values)

        ax.plot(t, ds["dvv"].values, label="ΔVV (ratio_images vv)")
        ax.plot(t, ds["dvh"].values, label="ΔVH (ratio_images vh)")
        ax.plot(t, ds["Rc"].values,  label="Rc (weighted)")

        ax.axhline(wet_thresh_db, linestyle="--", linewidth=1)

        wet = ds["wet"].values.astype(bool)
        ax.scatter(t[wet], ds["Rc"].values[wet], marker="x", s=35, label="wet (Rc<thresh)")

        ax.set_title(f"Orbit {o}")
        ax.set_ylabel("dB (if ratio_images is dB)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        ax.set_xlim(xmin_,xmax_)

    if title:
        fig.suptitle(title, y=0.99,fontweight = 'bold')
    plt.tight_layout()
    plt.show()
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

def plot_rc_timeseries(
    ax,
    ds,
    start,
    end,
    title,
    show_missing_annotation=False,
):
    """
    Plot ΔVV, ΔVH, and Rc for one orbit.
    """

    wet_thresh = -2.0

    ds_plot = ds.sel(Time=slice(start, end))

    time = pd.to_datetime(ds_plot.Time.values)

    # Plot available variables
    if np.isfinite(ds_plot["dvv"].values).any():
        ax.plot(
            time,
            ds_plot["dvv"].values,
            marker="o",
            ms=3,
            lw=1.5,
            label=r"$\Delta VV$",
        )

    if np.isfinite(ds_plot["dvh"].values).any():
        ax.plot(
            time,
            ds_plot["dvh"].values,
            marker="o",
            ms=3,
            lw=1.5,
            label=r"$\Delta VH$",
        )

    if np.isfinite(ds_plot["Rc"].values).any():
        ax.plot(
            time,
            ds_plot["Rc"].values,
            marker="o",
            ms=3,
            lw=2,
            label=r"$R_c$",
        )

    # Wet threshold
    ax.axhline(
        wet_thresh,
        color="black",
        ls="--",
        lw=1.5,
        label="Wet threshold",
    )

    ax.set_xlim(start, end)
    ax.set_title(title, fontweight="bold")

    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Explicitly communicate why Rc is missing
    if show_missing_annotation:
        ax.text(
            0.5,
            0.08,
            "VH unavailable\n$R_c$ cannot be calculated",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    return

def plot_phase_strip(
    ax,
    result,
    y,
    label,
    start,
    end,
    height=0.55,
):
    """
    Plot phase intervals from run_s1_rc_minima_year output.
    """

    # Use the same phase colors as the manuscript figures
    phase_colors = {
        "dry": "white",

        "moist": "#a9c4e4",
        "moistening": "#a9c4e4",

        "ripe": "#fdbb74",
        "ripening": "#fdbb74",

        "output": "#ff7f0e",
        "drain": "#ff7f0e",
        "draining": "#ff7f0e",
    }
    phases = result["phases"]

    starts = pd.to_datetime(phases["interval_start"].values)
    ends   = pd.to_datetime(phases["interval_end"].values)
    names  = phases["phase_name"].values

    for t0, t1, phase in zip(starts, ends, names):

        # Clip interval to plotting window
        left = max(pd.Timestamp(t0), start)
        right = min(pd.Timestamp(t1), end)

        if right <= left:
            continue

        ax.barh(
            y,
            width=(right - left).total_seconds() / 86400,
            left=mdates.date2num(left),
            height=height,
            color=phase_colors[str(phase)],
            edgecolor="none",
            align="center",
        )

    # Optional marker for sustained draining onset
    output_start = result.get("output_start", pd.NaT)

    if pd.notna(output_start):
        output_start = pd.Timestamp(output_start)

        if start <= output_start <= end:
            ax.plot(
                mdates.date2num(output_start),
                y,
                marker="|",
                color="black",
                ms=18,
                mew=2.2,
            )

# def _phase_to_time_series(ds: xr.Dataset, ds_phase: xr.Dataset, phase_var: str = None) -> xr.DataArray:
#     """
#     Convert ds_phase to a per-ds.time phase code DataArray.

#     Supports:
#       A) interval style: coords include interval_start + variable 'phase_id' and 'interval_end'
#       B) time style: ds_phase has a time dim and a phase variable (phase_var or inferred)
#     """
#     time = pd.to_datetime(ds["time"].values)

#     # --- A) interval-style (our classifier output) ---
#     if ("interval_start" in ds_phase.coords) and ("phase_id" in ds_phase.data_vars):
#         starts = pd.to_datetime(ds_phase["interval_start"].values)
#         ends   = pd.to_datetime(ds_phase["interval_end"].values)
#         codes  = ds_phase["phase_id"].values.astype(int)

#         # For each ds time, find which interval it belongs to: start <= t < end
#         # Efficient approach: find rightmost start <= t, then check t < end
#         idx = np.searchsorted(starts.values, time.values, side="right") - 1

#         phase = np.full(len(time), np.nan, dtype=float)
#         valid = (idx >= 0) & (idx < len(starts))
#         idxv = idx[valid]

#         # check end bound
#         in_interval = time.values[valid] < ends.values[idxv]
#         phase_idx = np.where(valid)[0][in_interval]
#         phase[phase_idx] = codes[idxv[in_interval]]

#         # Outside intervals -> NaN; you can choose to fill dry=0
#         phase = np.where(np.isnan(phase), 0, phase).astype(int)

#         return xr.DataArray(phase, dims=["time"], coords={"time": ds["time"]}, name="phase_code")

#     # --- B) time-style (SnowModel-like) ---
#     if phase_var is None:
#         # try common names
#         for cand in ["phase_code_3h", "phase_code", "phase_id", "phase"]:
#             if cand in ds_phase:
#                 phase_var = cand
#                 break
#     if phase_var is None:
#         raise ValueError("Could not infer phase variable in ds_phase. Pass phase_var='...'")

#     # Align to ds time
#     ph = ds_phase[phase_var]
#     if "time" not in ph.dims:
#         raise ValueError(f"Phase variable {phase_var} does not have a 'time' dimension.")

#     return ph.reindex(time=ds["time"], method="nearest")  # or .interp(time=..., method="nearest")


# def plot_snowmelt_phases_filltop(
#     ds,
#     ds_phase,
#     swe_var="swed",
#     runoff_var="roff",
#     title=None,
#     t_start=None,
#     t_end=None,
#     figsize=(14, 4),
#     ax=None,
#     ylim_=None,
#     phase_var=None,  # only needed for time-style ds_phase if not one of common names
# ):
#     """
#     Plot SWE and runoff with phases shaded from the SWE line up to the top of the axis.

#     Phase codes:
#       0 = dry (not shaded)
#       1 = moistening  (light blue)
#       2 = ripening    (light yellow)
#       3 = output      (light red)

#     ds_phase can be either:
#       - interval-style (interval_start/interval_end/phase_id) from SAR classifier
#       - time-style (phase_code_3h or similar) from SnowModel
#     """

#     # --- time selection / zoom ---
#     if t_start is not None or t_end is not None:
#         tsel = slice(t_start, t_end)
#         ds_ = ds.sel(time=tsel)
#     else:
#         ds_ = ds

#     time   = ds_["time"]
#     swe    = ds_[swe_var]

#     # Convert ds_phase -> per-time phase series aligned to ds_.time
#     ph_ts = _phase_to_time_series(ds_, ds_phase, phase_var=phase_var)

#     # Boolean masks by phase code
#     is_output   = (ph_ts == 3)
#     is_ripening = (ph_ts == 2)
#     is_moist    = (ph_ts == 1)

#     # Convert to numpy for fill_between
#     t_vals   = time.values
#     swe_vals = swe.values.astype(float)

#     # --- plotting ---
#     if ax is None:
#         fig, ax1 = plt.subplots(figsize=figsize)
#     else:
#         ax1 = ax

#     # SWE
#     ax1.plot(time, swe, "k-", linewidth=1.6, label="SWE (m)")

#     # Decide top of shading
#     swe_max = np.nanmax(swe_vals)
#     margin  = max(0.1, 0.2 * swe_max)
#     y_top   = swe_max + margin

#     if ylim_ is None:
#         ax1.set_ylim(bottom=0.0, top=y_top)
#     else:
#         ax1.set_ylim(bottom=0.0, top=ylim_)

#     def _shade_phase(mask_da, color, alpha):
#         m = mask_da.values.astype(bool)
#         if m.size == 0:
#             return
#         valid = m & np.isfinite(swe_vals)
#         if not np.any(valid):
#             return
#         ax1.fill_between(
#             t_vals,
#             swe_vals,
#             y_top if ylim_ is None else ylim_,
#             where=valid,
#             color=color,
#             alpha=alpha,
#             interpolate=True,
#         )

#     # Draw in order
#     _shade_phase(is_moist,    color="lightblue",     alpha=0.4)
#     _shade_phase(is_ripening, color="palegoldenrod", alpha=0.5)
#     _shade_phase(is_output,   color="lightcoral",    alpha=0.4)

#     ax1.set_ylabel("SWE (m)")
#     if title is not None:
#         ax1.set_title(title)
#     ax1.grid(alpha=0.3)

#     if ax is None:
#         fig.autofmt_xdate()

#     phase_patches = [
#         Patch(facecolor="lightcoral",    alpha=0.4, label="Output"),
#         Patch(facecolor="palegoldenrod", alpha=0.5, label="Ripening"),
#         Patch(facecolor="lightblue",     alpha=0.4, label="Moistening"),
#     ]

#     ax1.legend([p for p in phase_patches], [p.get_label() for p in phase_patches], loc="lower right")

#     if ax is None:
#         plt.tight_layout()
#         plt.show()
#         return
#     else:
#         return {"Output": phase_patches[0], "Ripening": phase_patches[1], "Moistening": phase_patches[2]}, (y_top if ylim_ is None else ylim_)
