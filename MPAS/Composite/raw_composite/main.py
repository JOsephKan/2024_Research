# This program is to composite the raw data
# %% section 1: environment setting
## import official packages
import os
import sys
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

## load self-defined package
sys.path.append('/home/b11209013/Package')
import DataProcess as dp         #type: ignore

## import system environment parameters
case: str = sys.argv[1]
#case: str = 'CNTL'

# %% section 1.5: functions
## load data
def load_data(
        path: str,
        var : str,
        lim
) -> np.ndarray:
    with nc.Dataset(f'{path}/{var}.nc', 'r') as f:
        data = f.variables[var][:, :, lim, :].mean(axis=2)

    return data

# %% section 2: Load data
## load compsiting events
comp_ref = np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/{case}.npy')

lon_ref : np.ndarray = comp_ref[0]
time_ref: np.ndarray = comp_ref[1]

## load MPAS data
mpas_path: str      = f'/work/b11209013/2024_Research/MPAS/merged_data/{case}/'

var_list: list[str] = ['theta', 'qv', 'q1', 'rthcuten', 'rqvcuten', 'rthratenlw', 'rthratensw']

### load dimension
with nc.Dataset(f'{mpas_path}/theta.nc', 'r') as f:
    lat : np.ndarray = f.variables['lat'][:]
    lon : np.ndarray = f.variables['lon'][:]
    lev : np.ndarray = f.variables['lev'][:]
    time: np.ndarray = f.variables['time'][:]

lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
lat = lat[lat_lim]

### load data
data: dict[str, np.ndarray] = {
    var: load_data(mpas_path, var, lat_lim)
    for var in var_list
}

data_ano: dict[str, np.ndarray] = {
    var: data[var] - data[var].mean(axis=(0, 2))[None, :, None]
    for var in var_list
}

ltime, llev, llon = data_ano['theta'].shape

# %% section 3: Composite
## designing the time interval
time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(time_ref.size)
]

time_ticks = np.linspace(-4, 4, 33)

## processing data
### unit conversion
theta2t = (1000/lev[None, :, None])**(-0.286)

data_convert = {
    't'        : data_ano['theta']*theta2t,
    'q1'       : data_ano['q1']*86400/1004.5,
    'qv'       : data_ano['qv']*1000,
    'rqvcuten' : data_ano['rqvcuten']*1000*86400,
    'rtcuten'  : data_ano["rthcuten"]*86400*theta2t,
    'rtratenlw': data_ano["rthratenlw"]*86400*theta2t,
    'rtratensw': data_ano["rthratensw"]*86400*theta2t
}

var_list = data_convert.keys() # update variable list

## composite
data_sel: dict[str, np.ndarray] = {
    var: np.array([
        data_convert[var][time_itv[i], :, lon_ref[i]]
        for i in range(time_ref.size)
    ]).mean(axis=0).T
    for var in (var_list)
}
# %% section: plot
heating_sum = data_sel['rtratenlw'] + data_sel['rtratensw'] + data_sel['rtcuten']

## vertical mean
q1_mean = np.sum(
    np.multiply((data_sel['q1'][1:]+data_sel['q1'][:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
)

t_mean = np.sum(
    np.multiply(((data_sel['t'][1:]+data_sel['t'][:-1])/2), (np.diff(lev)[..., None])*100),
    axis=0
) / np.sum(np.diff(lev)*100)

lw_mean = np.sum(
    np.multiply((data_sel['rtratenlw'][1:]+data_sel['rtratenlw'][:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)

sw_mean = np.sum(
    np.multiply((data_sel['rtratenlw'][1:]+data_sel['rtratensw'][:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)

cu_mean = np.sum(
    np.multiply((data_sel['rtcuten'][1:]+data_sel['rtcuten'][:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)


# %%
## plotting the vertical profile

plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

# First subplot with contourf
ax1 = plt.subplot(gs[0])
cr1 = ax1.contourf(
    time_ticks, lev[:-2], heating_sum[:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax1.contour(
    time_ticks, lev[:-2], data_sel['t'][:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = plt.contour(
    time_ticks, lev[:-2], data_sel['qv'][:-2],
    levels=[-0.4, -0.3, -0.2,  0.2, 0.3, 0.4],
    colors='forestgreen', linewidths=1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
ax1.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
ax1.set_xlim(4, -4)
ax1.set_ylim(1000, 100)
ax1.set_ylabel("Level [hPa]")
plt.clabel(c1, inline=True, fontsize=8)

# Second subplot with line plot
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(
    time_ticks, lw_mean,
    color='royalblue',
    label='LW'
)
ax2.plot(
    time_ticks, sw_mean,
    color='sienna',
    label='SW'
)
ax2.plot(
    time_ticks, cu_mean,
    color='forestgreen',
    label='Cu'
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax2.set_xlim(4, -4)
ax2.set_ylim(-4, 5)
ax2.set_xticks(np.linspace(4, -4, 9))
ax2.set_yticks([-3, -1.5, 0, 1.5, 3])
ax2.set_ylabel('K/day')

cax = inset_axes(ax1, width="1.5%", height="80%", loc="center right", bbox_to_anchor=(0.04, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)


ax3 = ax2.twinx()
ax3.plot(
    time_ticks, t_mean,  # Plot data along the first row as an example
    color='k', 
    label='T'
)
plt.gca().spines['top'].set_visible(False)
ax3.set_ylabel('K')
ax3.set_ylim(-0.34, 0.34)
ax3.set_yticks(np.linspace(-0.4, 0.4, 5))

plt.text(2.3, -0.8, 'Day After')
plt.text(-1.7, -0.8, 'Day Before')

cbar = fig.colorbar(
    cr1, cax=cax, ax=[ax1, ax2], location="right",  label="K/day")
cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6, 8])

ax1.set_title(f'Case: {case}\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen)',
fontsize=10, loc='left')

plt.savefig(f'{case}_raw_composite.png', dpi=300)