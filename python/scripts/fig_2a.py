# fig_2a.py
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_snowmelt_phase_comp(year: int,
                             cues_sm_biased_swe: xr.DataArray = None,
                             cues_sm_corrected_swe: xr.DataArray = None,
                             cues_sm_optimized_swe: xr.DataArray = None,
                             cues_obs_no_qa_df: pd.DataFrame = None,
                             cues_obs_qa_df: pd.DataFrame = None,
                             start_date: str = None,
                             end_date: str = None,
                             ax=None,
                             ds_temp_day: xr.Dataset = None,
                             plotTitle: bool = True,
                             ymax_: float = None,
                             plotInsetTitle: bool = False,
                             ):
    if start_date is None:
        start_date = np.datetime64(f'{year-1}-10-01')
    else:
        start_date = np.datetime64(start_date)

    if end_date is None:
        end_date = np.datetime64(f'{year}-10-01')
    else:
        end_date = np.datetime64(end_date)

    if cues_sm_biased_swe is not None:
        cues_sm_biased_swe_yr = cues_sm_biased_swe.where(cues_sm_biased_swe.time >= start_date,drop = True) \
                                                   .where(cues_sm_biased_swe.time <= end_date,drop = True)
        sm_biased_da = cues_sm_biased_swe_yr.resample(time='1D').mean()[:,0,0]
    if cues_sm_corrected_swe is not None:
        cues_sm_corrected_swe_yr = cues_sm_corrected_swe.where(cues_sm_corrected_swe.time >= start_date,drop = True) \
                                                   .where(cues_sm_corrected_swe.time <= end_date,drop = True)
        sm_corrected_da = cues_sm_corrected_swe_yr.resample(time='1D').mean()[:,0,0]
    if cues_sm_optimized_swe is not None:
        cues_sm_optimized_swe_yr = cues_sm_optimized_swe.where(cues_sm_optimized_swe.time >= start_date,drop = True) \
                                                   .where(cues_sm_optimized_swe.time <= end_date,drop = True)
        sm_optimized_da = cues_sm_optimized_swe_yr.resample(time='1D').mean()[:,0,0]
    if year <= 2017:
        cues_obs_df_yr = cues_obs_qa_df[(cues_obs_qa_df['DateTime'] >= start_date) & (cues_obs_qa_df['DateTime'] <= end_date)]    
    else:
        cues_obs_df_yr = cues_obs_no_qa_df[(cues_obs_no_qa_df['DateTime'] >= start_date) & (cues_obs_no_qa_df['DateTime'] <= end_date)]
    
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (10,5),dpi = 200)
    if cues_sm_biased_swe is not None:
        ax.plot(sm_biased_da.time.values,sm_biased_da.values,label = 'SM-baseline')
    if cues_sm_corrected_swe is not None:
        ax.plot(sm_corrected_da.time.values,sm_corrected_da.values,label = 'SM-prec')
    if cues_sm_optimized_swe is not None:
        ax.plot(sm_optimized_da.time.values,sm_optimized_da.values,label = 'SM-met')
    ax.plot(cues_obs_df_yr.DateTime.values,cues_obs_df_yr.SWE.values,label = 'Observed SWE',color = 'black')
    # sm_biased_da.plot(ax=ax,label = 'Modeled - Biased')
    # sm_corrected_da.plot(ax=ax,label = 'Modeled - Corrected')
    # cues_obs_df_yr.plot(ax=ax,x='DateTime',y='SWE',label = 'Observed',color = 'black',linestyle = '--')
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                      frameon=True, fontsize=8, ncol=5)
    if plotTitle:
        ax.set_title('a) Models and Observed Data',fontweight = 'bold')
    if plotInsetTitle:
        ax.text(0.01, 0.97, 'a) Models and Observed Data', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left')
    ax.set_ylabel('SWE\n[mm]',fontweight = 'bold')
    ax.set_xlabel(None)
    ax.grid(which='major', axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.5, linewidth = 1)
    ax.set_ylim(0,ymax_)
    
    if ds_temp_day is not None:
        temp_slice = ds_temp_day.where(ds_temp_day['datetime'] >= start_date, drop=True) \
                                .where(ds_temp_day['datetime'] <= end_date, drop=True)
        ax2 = ax.twinx()
        ax2.plot(temp_slice['datetime'].values, temp_slice['air_temperature'].values,
                 label='Observed Air Temp', color='purple', alpha=0.5, linewidth=0.8)
        ax2.set_ylabel('Air Temperature\n[°C]', fontweight='bold', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.axhline(y=0, color='purple', linestyle='--', linewidth=1, alpha=0.8)
        # Merge temp legend into main legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=8, ncol=5)
    
    return
    # return sm_biased_da,sm_corrected_da,sm_optimized_da,cues_obs_df_yr

def plot_lysimeter(year: int,
                   df_lys: pd.DataFrame,
                   start_date: str = None,
                   end_date: str = None,
                   ax=None,
                   plotTitle: bool = True,
                   plotInsetTitle: bool = False,
                   ):
    
    if start_date is None:
        start_date = np.datetime64(f'{year-1}-10-01')
    else:
        start_date = np.datetime64(start_date)

    if end_date is None:
        end_date = np.datetime64(f'{year}-10-01')
    else:
        end_date = np.datetime64(end_date)
    plotCols_S = ['tb1_south','tb2_south','tb3_south','tb4_south']
    plotCols_N = ['tb5_north','tb6_north','tb7_north','tb8_north']

    df_lys['datetime'] = pd.to_datetime(df_lys['datetime'])
    df_lys_yr = df_lys[df_lys['datetime'].dt.year == year]
    df_lys_yr = df_lys[(df_lys['datetime'] >= start_date) & (df_lys['datetime'] <= end_date)]
    # df_lys_yr = df_lys_yr.set_index('datetime').resample('D').sum().reset_index()
    if ax is None:
        fig,ax = plt.subplots(figsize = (10,5),dpi = 200)
    count = 0
    for lys in plotCols_N:
         if count == 0:
             ax.plot(df_lys_yr['datetime'].values, df_lys_yr[lys].values, label='Grouping 1', color='C3', alpha=0.5)
         else:
             ax.plot(df_lys_yr['datetime'].values, df_lys_yr[lys].values, label='', color='C3', alpha=0.5)
         count += 1
    count = 0
    for lys in plotCols_S:
        if count == 0:
            ax.plot(df_lys_yr['datetime'].values, df_lys_yr[lys].values, label='Grouping 2', color='C4', alpha=0.5)
        else:
            ax.plot(df_lys_yr['datetime'].values, df_lys_yr[lys].values, label='', color='C4', alpha=0.5)
        count += 1
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      frameon=True, fontsize=8)
    ax.grid(which='major', axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.5, linewidth = 1)
    if plotTitle:
        ax.set_title('b) Lysimeter Melt',fontweight = 'bold')
    if plotInsetTitle:
        ax.text(0.01, 0.97, 'b) Lysimeter Melt', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left')
    ax.set_ylabel('Melt water eq.\n[mm]',fontweight = 'bold')
    ax.set_xlabel(None)
    return



def find_meltout_date(df, swe_col="SWE", threshold=0.01):
    """
    Find first date after peak SWE when SWE drops below threshold.
    """
    s = df[swe_col].copy()
    s.index = pd.to_datetime(df.index)

    # Find peak SWE date
    peak_date = s.idxmax()

    # Only look after peak
    after_peak = s.loc[peak_date:]

    # First day below threshold
    meltout = after_peak[after_peak <= threshold]

    if len(meltout) == 0:
        return None

    return meltout.index[0]

def identify_swe_meltout_date(year:int,
                              cues_obs_no_qa_df: pd.DataFrame,
                              cues_obs_qa_df: pd.DataFrame,
                              ):
    if year <= 2017:
        cues_obs_df_yr = cues_obs_qa_df[(cues_obs_qa_df['DateTime'] >= np.datetime64(f'{year-1}-10-01')) & (cues_obs_qa_df['DateTime'] < np.datetime64(f'{year}-10-01'))]    
    else:
        cues_obs_df_yr = cues_obs_no_qa_df[(cues_obs_no_qa_df['DateTime'] >= np.datetime64(f'{year-1}-10-01')) & (cues_obs_no_qa_df['DateTime'] < np.datetime64(f'{year}-10-01'))]
    meltout_date = find_meltout_date(cues_obs_df_yr.set_index('DateTime'), swe_col="SWE", threshold=0.01)
    return np.array(meltout_date,dtype = 'datetime64[ns]')


def make_panel_c_daily(
    ph_sm_biased,
    ph_sm_corrected,
    ph_sm_biased_wt=None,
    ph_sm_optimized=None,
    ph_s1=None,
    ph_s1_2=None,
    start="2017-01-01",
    end="2017-07-01",
    method_labels=None,
    method_order=None,
    ax=None,
    phase_colors=None,
    plotTitle: bool = False,
    plotInsetTitle: bool = False,
):
    """
    Panel C: daily melt-phase timeline for multiple methods.
    Each method is one horizontal row; days are coloured by phase.

    Parameters
    ----------
    ph_sm_biased, ph_sm_corrected : xr.Dataset
        SnowModel phase datasets (3-hourly).
    ph_sm_biased_wt, ph_sm_optimized : xr.Dataset, optional
        Additional SnowModel variants.
    ph_s1, ph_s1_2 : dict, optional
        Sentinel-1 output dicts with 'qc_ok' and 'phases' keys.
    start, end : str
        Date window (YYYY-MM-DD).
    method_labels : dict, optional
        {internal_name: display_label} mapping.
    method_order : tuple of str, optional
        Order of rows (top to bottom) using internal names.
    ax : matplotlib Axes, optional
    phase_colors : dict, optional
        {phase_name: color}. Defaults provided.

    Returns
    -------
    phases_dict : dict
        {internal_name: pd.Series} of daily phase series.
    """
    if phase_colors is None:
        phase_colors = {
            "dry":         "white",
            "moistening":  "#AEC7E8",   # light blue
            "ripening":    "#FFBB78",   # light orange
            "output":      "#FF7F0E",   # orange
            "na":          "lightgray",
        }

    if method_labels is None:
        method_labels = {
            "SM corrected":  "Model-precip-adj",
            "SM biased":     "Model-raw",
            "SM biased wt":  "Model-wt-adj",
            "SM optimized":  "Model-optimized",
            "S1 method 1":   "S1 VV",
            "S1 method 2":   "S1 weighted",
        }

    # Build internal-name → object mapping
    source_map = {}
    if ph_sm_biased is not None:
        source_map["SM biased"] = ("sm", ph_sm_biased)
    if ph_sm_corrected is not None:
        source_map["SM corrected"] = ("sm", ph_sm_corrected)
    if ph_sm_biased_wt is not None:
        source_map["SM biased wt"] = ("sm", ph_sm_biased_wt)
    if ph_sm_optimized is not None:
        source_map["SM optimized"] = ("sm", ph_sm_optimized)
    if ph_s1 is not None:
        source_map["S1 method 1"] = ("s1", ph_s1)
    if ph_s1_2 is not None:
        source_map["S1 method 2"] = ("s1", ph_s1_2)

    # Convert all to daily phase series
    phases_dict = {}
    for name, (kind, obj) in source_map.items():
        phases_dict[name] = to_daily_phase_series(obj, kind, start, end)

    # Determine row order
    if method_order is None:
        method_order = list(phases_dict.keys())
    else:
        method_order = [m for m in method_order if m in phases_dict]

    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    n_methods = len(method_order)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 0.5 * n_methods + 0.5), dpi=200)

    # Map phases to integers for imshow
    phase_list = ["dry", "moistening", "ripening", "output", "na"]
    phase_to_int = {p: i for i, p in enumerate(phase_list)}
    cmap_colors = [phase_colors.get(p, "lightgray") for p in phase_list]
    cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)
    bounds = np.arange(len(phase_list) + 1) - 0.5
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Build 2D array (methods × days)
    arr = np.full((n_methods, len(dates)), phase_to_int["na"])
    for row_i, m in enumerate(method_order):
        s = phases_dict[m].reindex(dates).fillna("na")
        for col_j, d in enumerate(dates):
            val = str(s.iloc[col_j]).lower().strip()
            arr[row_i, col_j] = phase_to_int.get(val, phase_to_int["na"])

    ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[
            np.datetime64(start),
            np.datetime64(end),
            n_methods - 0.5,
            -0.5,
        ],
    )

    # Y-axis labels
    display_labels = [method_labels.get(m, m) for m in method_order]
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(display_labels, fontsize=8)
    if plotTitle:
        ax.set_title("e) Melt Phase Classification", fontweight="bold")
    if plotInsetTitle:
        ax.text(0.01, 0.97, "e) Melt Phase Classification", transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left', zorder=5)
    ax.set_xlabel('Date',fontweight = 'bold')
    ax.set_ylabel('Method',fontweight = 'bold')
    ax.grid(which="major", axis="x", linestyle="-", color="gray", alpha=0.3, linewidth=0.8)
    ax.set_yticks(np.arange(n_methods + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', length=0)
    ax.grid(which='minor', axis='y', linestyle='--', color='gray', alpha=0.5, linewidth=1)
    ax.tick_params(axis='x', labelrotation=45)

    # Legend
    from matplotlib.patches import Patch
    phase_list_2 = ["dry", "moistening", "ripening", "draining", "na"]
    phase_colors_2 = {
            "dry":         "white",
            "moistening":  "#AEC7E8",   # light blue
            "ripening":    "#FFBB78",   # light orange
            "draining":      "#FF7F0E",   # orange
            "na":          "lightgray",
        }
    legend_elements = [
        Patch(facecolor=phase_colors_2[p], edgecolor="k", linewidth=0.5, label=p.capitalize())
        for p in phase_list_2 if p != "na"
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=8,
        title="Phase",
    )

    return phases_dict

def plot_vv_snow_identification_combined(sentinel_bscatter: xr.Dataset,
                                         sentinel_reference: xr.Dataset,
                                         vv_output: dict = None,
                                         band_name: str = 'vv',
                                         orbit_list: list = None,
                                         start_date: str = None,
                                         end_date: str = None,
                                         ax=None,
                                         title: str = None,
                                         idy: int = None,
                                         idx: int = None,
                                         plotTitle: bool = False,
                                         ymin_: float = None,
                                         ymax_: float = None,
                                         plotInsetTitle: bool = False,
                                         ):
    """
    Plot backscatter for a chosen polarisation band, grouped by orbit,
    with reference lines and wet-detection x-markers.

    Parameters
    ----------
    sentinel_bscatter : xr.Dataset
        Backscatter dataset.  Expected variable ``bscatter`` with dims
        (time, band, y, x) and coords ``band`` = ['vh','vv'],
        ``sat:relative_orbit`` along time.
    sentinel_reference : xr.Dataset
        Reference backscatter dataset with variable ``reference_bscatter``
        and dims (sat:relative_orbit, band, y, x).
    vv_output : dict, optional
        Not currently used (reserved for future overlay).
    band_name : str
    orbit_list : list of int, optional
        Which relative orbits to plot, e.g. ``[137, 144]``.
        If *None*, all orbits present in the data are plotted.
    start_date, end_date : str
        Date range (YYYY-MM-DD).
    ax : matplotlib.axes.Axes
        Axis to plot on (created if None).
    title : str
        Plot title.  Defaults to ``'{BAND} Backscatter …'``.
    idy, idx : int
        Spatial indices for point extraction via ``.isel(y=, x=)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    if title is None:
        title = f"{band_name.upper()} Backscatter & Reference (by orbit)"

    # ---- colour / label / marker per orbit number ----
    orbit_style = {
        144: {'color': 'C5', 'marker': 'o', 'label': 'asc_144 (AM)'},
        137: {'color': 'C8', 'marker': 's', 'label': 'dsc_137 (PM)'},
         64: {'color': 'C2', 'marker': 'D', 'label': 'dsc_64 (PM)'},
    }
    _fallback_colors = ['C3', 'C4', 'C5', 'C6']

    # Helper: derive AM/PM suffix from orbit_style labels
    def _am_pm(orbit_num):
        style_label = orbit_style.get(int(orbit_num), {}).get('label', '')
        return 'AM' if '(AM)' in style_label else 'PM'

    # ---- select the requested polarisation band ----
    bsc = sentinel_bscatter['bscatter']
    if 'band' in bsc.dims and 'band' in sentinel_bscatter.coords:
        bsc = bsc.sel(band=band_name)              # (time, y, x)
    # extract point
    if idy is not None and idx is not None:
        bsc = bsc.isel(y=idy, x=idx)               # (time,)

    # extract orbit coordinate and unique orbit numbers
    orbits_coord = sentinel_bscatter['sat:relative_orbit']
    unique_orbits = np.unique(orbits_coord.values)

    if orbit_list is not None:
        unique_orbits = np.array([o for o in unique_orbits if int(o) in orbit_list])

    wet_thresh_db = -2  # Marin et al. threshold

    # ---- single loop: fetch reference, compute drop, plot ----
    plotted_orbits = {}
    for i, orbit_num in enumerate(sorted(unique_orbits)):
        orbit_mask = orbits_coord.values == orbit_num
        times  = bsc.time.values[orbit_mask]
        values = bsc.values[orbit_mask].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            values_db = 10 * np.log10(values)
        valid = ~np.isnan(values_db)
        if not np.any(valid):
            continue

        # fetch reference value for this orbit
        ref_val_db = None
        try:
            ref_val = sentinel_reference['reference_bscatter'].sel(
                **{'sat:relative_orbit': orbit_num, 'band': band_name}
            )
            if idy is not None and idx is not None:
                ref_val = ref_val.isel(y=idy, x=idx)
            ref_val_db = 10 * np.log10(float(ref_val.values))
        except Exception:
            pass

        # compute drop from reference; fall back to raw dB if reference unavailable
        drop_db = values_db[valid] - ref_val_db if ref_val_db is not None else values_db[valid]

        style = orbit_style.get(int(orbit_num),
                                {'color': _fallback_colors[i % len(_fallback_colors)],
                                 'marker': '^',
                                 'label': f'orbit_{orbit_num}'})
        ax.plot(times[valid], drop_db,
                linestyle='-', marker=style['marker'],
                color=style['color'], label=f'Orbit-{orbit_num}\n(Δσ{_am_pm(orbit_num)})',
                markersize=5, alpha=0.7)

        plotted_orbits[int(orbit_num)] = {
            'color': style['color'],
            'label': style['label'],
            'times': times[valid],
            'drop_db': drop_db,
            'ref_val_db': ref_val_db,
        }

    # ---- wet x-markers ----
    any_wet = False
    for orbit_num, info in plotted_orbits.items():
        if info['ref_val_db'] is None:
            continue
        wet = info['drop_db'] < wet_thresh_db
        if np.any(wet):
            ax.scatter(info['times'][wet], info['drop_db'][wet],
                       marker='x', s=80, color='red', zorder=6, linewidths=2)
            any_wet = True

    # ---- drop threshold line ----
    ax.axhline(wet_thresh_db, color='black', linestyle='--', linewidth=1.5)
    threshold_line = plt.matplotlib.lines.Line2D([], [], color='black', linestyle='--',
                                                 linewidth=1.5, label='drop threshold')

    # ---- axes formatting ----
    if start_date is not None and end_date is not None:
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    if plotTitle:
        ax.set_title(title, fontweight='bold')
    if plotInsetTitle:
        ax.text(0.01, 0.97, title, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left')
    ax.set_ylabel('Backscatter\nAnomaly [Δσ; dB]', fontweight='bold')
    ax.set_xlabel(None, fontweight='bold')
    ax.set_ylim(ymin_,ymax_)

    # orbit legend (outside, upper right)
    main_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.9),
                            frameon=True, fontsize=8, title="Backscatter")
    ax.add_artist(main_legend)

    # threshold + wet markers legend (bottom left, no title)
    bottom_handles = [threshold_line]
    if any_wet:
        bottom_handles.append(
            plt.matplotlib.lines.Line2D([], [], linestyle='None', marker='x',
                                        color='red', markersize=7, markeredgewidth=2,
                                        label='Wet')
        )
    ax.legend(handles=bottom_handles, loc="lower left", frameon=True, fontsize=8)

    ax.grid(which='major', axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.5, linewidth=1)
    return ax


DEFAULT_PHASE_MAP = {
    "moist": "moistening",
    "moisten": "moistening",
    "ripe": "ripening",
    "ripen": "ripening",
    "wet": "output",
    "melt": "output",
    "nan": "na",
    "none": "na",
}
def sm_3h_to_daily_majority(ds, phase_var="phase_name_3h") -> pd.Series:
    s = ds[phase_var].to_series()
    s.index = pd.to_datetime(s.index)
    daily = s.groupby(s.index.normalize()).agg(
        lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else np.nan
    )
    daily.name = "phase"
    return daily

def s1_intervals_to_daily_fill(ds, start, end,
                              start_var="interval_start",
                              end_var="interval_end",
                              phase_var="phase_name",
                              fill_value="na") -> pd.Series:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    idx = pd.date_range(start, end, freq="D")

    starts = pd.to_datetime(ds[start_var].values)
    ends   = pd.to_datetime(ds[end_var].values)
    phases = pd.Series(ds[phase_var].values).values

    order = np.argsort(starts.values)
    starts, ends, phases = starts[order], ends[order], phases[order]

    out = pd.Series(index=idx, dtype="object")
    for st, en, ph in zip(starts, ends, phases):
        if pd.isna(st) or pd.isna(en):
            continue
        d0 = max(st.normalize(), start)
        d1 = min(en.normalize(), end)
        if d1 >= d0:
            out.loc[d0:d1] = ph

    out.name = "phase"
    return out.fillna(fill_value)

def normalize_phase_series(s: pd.Series, mapping=None, fill_value="na") -> pd.Series:
    if mapping is None:
        mapping = DEFAULT_PHASE_MAP
    s2 = s.astype("object").apply(lambda v: str(v).lower().strip() if pd.notna(v) else np.nan)
    s2 = s2.replace(mapping).fillna(fill_value)
    return s2

def to_daily_phase_series(obj, kind, start, end,
                          sm_phase_var="phase_name_3h",
                          s1_start_var="interval_start",
                          s1_end_var="interval_end",
                          s1_phase_var="phase_name",
                          mapping=None):
    """
    Convert xarray Dataset or Sentinel dictionary -> daily pd.Series of normalized phases.
    kind: "sm" or "s1"
    
    For S1 (Sentinel) data, accepts either:
    - Dictionary with 'qc_ok' and 'phases' keys (new format)
    - xarray Dataset directly (legacy format)
    
    If dict and qc_ok=False, returns NaN-filled series.
    """
    ALLOWED = {"dry", "moistening", "ripening", "output", "na"}
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    
    # Handle Sentinel dictionary format with QC checking
    if kind == "s1" and isinstance(obj, dict):
        if not obj.get('qc_ok', False):
            # QC failed: return NaN-filled series (all "na")
            return pd.Series("na", index=dates, name="phase")
        # QC passed: extract phases dataset
        obj = obj['phases']

    if kind == "sm":
        s = sm_3h_to_daily_majority(obj, phase_var=sm_phase_var).reindex(dates)
    elif kind == "s1":
        s = s1_intervals_to_daily_fill(
            obj, start, end,
            start_var=s1_start_var, end_var=s1_end_var, phase_var=s1_phase_var
        ).reindex(dates)
    else:
        raise ValueError("kind must be 'sm' or 's1'")

    s = normalize_phase_series(s, mapping=mapping).fillna("na")

    # sanity: ensure no unknown labels
    u = set(pd.unique(s.astype(str).str.lower().str.strip()))
    unknown = sorted(u - ALLOWED)
    if unknown:
        raise ValueError(f"Unknown phase labels after normalization: {unknown}")

    return s

def plot_orbit_timeseries_subplot(ts_by_orbit,
                        wet_thresh_db=-2.0,
                        orbit_lst=None,
                        ax=None,
                        onlyRc=False,
                        title="",
                        plotTitle: bool = False,
                        plotInsetTitle: bool = False,
                        ymin_: float = None,
                        ymax_: float = None,
                        ):
    # ---- colour / marker / label per orbit (matches plot_vv_snow_identification_combined) ----
    orbit_style = {
        144: {'color': 'C5', 'marker': 'o', 'label_suffix': 'AM'},
        137: {'color': 'C8', 'marker': 's', 'label_suffix': 'PM'},
         64: {'color': 'C2', 'marker': 'D', 'label_suffix': 'PM'},
    }
    _fallback_colors = ['C3', 'C4', 'C5', 'C6']

    def _style(o):
        return orbit_style.get(int(o),
                               {'color': _fallback_colors[int(o) % len(_fallback_colors)],
                                'marker': '^', 'label_suffix': ''})

    orbits = [o for o, ds in ts_by_orbit.items() if ds is not None]
    if orbit_lst is not None:
        orbits = [o for o in orbits if o in orbit_lst]
    n = len(orbits)

    if ax is None:
        # Standalone mode: one sub-panel per orbit
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), dpi=160, sharex=True)
        if n == 1:
            axes = [axes]
        for ax_i, o in zip(axes, orbits):
            st = _style(o)
            ds = ts_by_orbit[o]
            t = pd.to_datetime(ds["Time"].values)
            if not onlyRc:
                ax_i.plot(t, ds["dvv"].values, linestyle='-', marker=st['marker'],
                          color=st['color'], alpha=0.4, markersize=4,
                          label=f"{o}-{st['label_suffix']} ΔVV")
                ax_i.plot(t, ds["dvh"].values, linestyle='--', marker=st['marker'],
                          color=st['color'], alpha=0.4, markersize=4,
                          label=f"{o}-{st['label_suffix']} ΔVH")
            ax_i.plot(t, ds["Rc"].values, linestyle='-', marker=st['marker'],
                      color=st['color'], markersize=5, alpha=0.7,
                      label=f"{o}-{st['label_suffix']} Rc")
            ax_i.axhline(wet_thresh_db, linestyle="--", linewidth=1, color="gray")
            wet = ds["wet"].values.astype(bool)
            ax_i.scatter(t[wet], ds["Rc"].values[wet], marker="x", s=80,
                         color='red', zorder=6, linewidths=2)
            ax_i.set_title(f"Orbit {o}")
            ax_i.set_ylabel("dB")
            ax_i.grid(True, alpha=0.25)
            ax_i.legend(loc="best", fontsize=8)
        ax_i.set_ylim(ymin_,ymax_)
        if title:
            if plotTitle:
                fig.suptitle(title, y=0.99)
        plt.tight_layout()
        plt.show()
        return axes
    else:
        # Embedded mode: overlay all orbits on the provided single axis
        any_wet = False
        for o in orbits:
            st = _style(o)
            ds = ts_by_orbit[o]
            t = pd.to_datetime(ds["Time"].values)
            if not onlyRc:
                ax.plot(t, ds["dvv"].values, linestyle='-', marker=st['marker'],
                        color=st['color'], alpha=0.4, markersize=4,
                        label=f"{o}-{st['label_suffix']} ΔVV")
                ax.plot(t, ds["dvh"].values, linestyle='--', marker=st['marker'],
                        color=st['color'], alpha=0.4, markersize=4,
                        label=f"{o}-{st['label_suffix']} ΔVH")
            ax.plot(t, ds["Rc"].values, linestyle='-', marker=st['marker'],
                    color=st['color'], markersize=5, alpha=0.7,
                    label=f"Orbit-{o}\n(Δσ{st['label_suffix']})")
            wet = ds["wet"].values.astype(bool)
            if np.any(wet):
                ax.scatter(t[wet], ds["Rc"].values[wet], marker="x", s=80,
                           color='red', zorder=6, linewidths=2)
                any_wet = True
        ax.axhline(wet_thresh_db, linestyle="--", linewidth=1.5, color="black")
        threshold_line = plt.matplotlib.lines.Line2D([], [], color='black', linestyle='--',
                                                     linewidth=1.5, label='drop threshold')
        ax.set_ylabel("Backscatter\nAnomaly [Δσ; dB]", fontweight='bold')
        ax.set_ylim(ymin_,ymax_)
        ax.grid(which='major', axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
        ax.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # orbit legend (outside, upper left)
        main_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                                borderaxespad=0, frameon=True, fontsize=8, title="Backscatter")
        ax.add_artist(main_legend)

        # threshold + wet markers legend (bottom left, no title)
        bottom_handles = [threshold_line]
        if any_wet:
            bottom_handles.append(
                plt.matplotlib.lines.Line2D([], [], linestyle='None', marker='x',
                                            color='red', markersize=7, markeredgewidth=2,
                                            label='Wet')
            )
        ax.legend(handles=bottom_handles, loc="lower left", frameon=True, fontsize=8)

        if title:
            if plotTitle:
                ax.set_title(title, fontweight='bold')
            if plotInsetTitle:
                ax.text(0.01, 0.97, title, transform=ax.transAxes,
                        fontsize=9, fontweight='bold', va='top', ha='left')
        return ax