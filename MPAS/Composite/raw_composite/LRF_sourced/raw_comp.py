# This program is to composite the raw data
# %% section 1: environment setting
## import package
from math import exp
import os
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc
from matplotlib import pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

## load self-defined package
sys.path.append('/home/b11209013/Package')
import DataProcess as dp         #type: ignore

## import system environment parameters
exp: str = sys.argv[1] # load experiment name
#exp: str = 'CNTL'

# %% section 1.5: functions
## load data
def load_data(
        path: str,
        var : str,
        lim
) -> np.ndarray:
    with nc.Dataset(f'{path}/{var}.nc', 'r') as f:
        data = f.variables[var][:, :, lim, :].mean(axis=2)

    return var, data

## vertical averaged
def vert_int(
        data: np.ndarray,
        lev : np.ndarray
) -> np.ndarray:
    
    data_ave: np.ndarray = (data[1:] + data[:-1])/2
    data_int: np.ndarray = np.sum(
        np.multiply(data_ave, np.diff(lev)[:, None]*100),
        axis = 0
    ) / np.sum(np.diff(lev)*100)
    
    return data_int

# %% section 2: Load data
## load compsiting events
lon_ref, time_ref = np.load(
    f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/Q1_sel/{exp}.npy'\
    )

## load MPAS data
mpas_path: str = f'/work/b11209013/2024_Research/MPAS/merged_data/{exp}/'

var_list: list[str] = ['theta', 'qv', 'q1']

### load dimension
dims: dict[str, np.ndarray] = dict()
with nc.Dataset(f'{mpas_path}{var_list[0]}.nc', 'r') as f:

    for key in f.dimensions.keys():
        dims[key] = f.variables[key][:]

lat_lim: list[int] = np.where(
    (dims['lat'] >= -5) & (dims['lat'] <= 5)
    )[0]

lat: np.ndarray = dims['lat'][lat_lim]
### load data
data: dict[str, np.ndarray] = dict(
    jl.Parallel(n_jobs = -1)(
        jl.delayed(load_data)(mpas_path, var, lat_lim)
        for var in var_list
    )
)

ltime, llev, llon = data[var_list[0]].shape

## load LRF file
lrf_data = jl.load(f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/LRF_{exp}.joblib')

lrf: dict[str, np.ndarray] = {
    var: np.where(np.isnan(lrf_data[var]), 0, lrf_data[var])
    for var in lrf_data.keys()
}

with nc.Dataset(f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/{exp}.nc', "r") as f:
    lrf: dict[str, np.ndarray] = {
        var: np.where(np.isnan(f.variables[var][:]), 0, f.variables[var][:])
        for var in ["lw", "sw", "cu", "tot"]
    }


# %% section 3: Processing data
## 3.1 convert unit
theta2t = lambda theta: theta*(1000/dims['lev'][None, :, None])**-0.286

data_convert: dict[str, np.ndarray] = {
    't' : theta2t(data['theta']),
    'q1': data['q1']*86400/1004.5,
    'qv': data['qv']*1000,
}

## 3.2 compute anomaly
data_ano: dict[str, np.ndarray] = {
    key: data_convert[key] - data_convert[key].mean(axis = (0, 2))[None, :, None]
    for key in data_convert.keys()
}

# %% section 4: Composite
## 4.1 designing the time interval
time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(time_ref.size)
]

time_ticks = np.linspace(-4, 4, 33)

## 4.2 composite
data_sel: dict[str, np.ndarray] = {
    key: np.array([
        data_ano[key][time_itv[i], :, lon_ref[i]]
        for i in range(time_ref.size)
    ]).mean(axis=0).T
    for key in data_ano.keys()
}

## 4.3 construct state vector
state_vec: np.ndarray = np.concatenate([data_sel['t'], data_sel['qv']], axis=0)


## 4.4 compute the heating
data_sel['cu'] = np.array(lrf['cu']) @ state_vec
data_sel['lw'] = np.array(lrf['lw']) @ state_vec
data_sel['sw'] = np.array(lrf['sw']) @ state_vec

data_sel['tot'] = data_sel['cu'] + data_sel['lw'] + data_sel['sw']

## vertical averaged
data_vint: dict[str, np.ndarray] = {
    key: vert_int(data_sel[key], dims['lev'])
    for key in data_sel.keys()
}

# %% section 5: Plot
## 5.1 plot setting
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

lev_cond = np.argmin(np.abs(dims['lev'] - 200))

## 5.2 plot
fig = plt.figure(figsize=(12, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

# First subplot with contourf
ax1 = plt.subplot(gs[0])
cr1 = ax1.contourf(
    time_ticks, dims['lev'][:lev_cond+1], data_sel['tot'][:lev_cond+1],
    cmap='RdBu_r',
    levels=np.linspace(-6, 10),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax1.contour(
    time_ticks, dims['lev'][:lev_cond+1], data_sel['t'][:lev_cond+1],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = plt.contour(
    time_ticks, dims['lev'][:lev_cond+1], data_sel['qv'][:lev_cond+1],
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
    time_ticks, data_vint['lw'],  
    color='royalblue',
    label='LW'
)
ax2.plot(
    time_ticks, data_vint['sw'],
    color='sienna',
    label='SW'
)
ax2.plot(
    time_ticks, data_vint['cu'],
    color='forestgreen',
    label='Cu'
)
ax2.plot(
    time_ticks, data_vint['tot'],
    color='k',
    linestyle='--',
    label='Tot'
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax2.set_xticks(np.linspace(4, -4, 9))
ax2.set_yticks([-4, -2, 0, 2, 4, 6])
ax2.set_xlim(4, -4)
ax2.set_ylim(-4, 6)
ax2.set_ylabel('K/day')

cax = inset_axes(ax1, width="1.5%", height="80%", loc="center right", bbox_to_anchor=(0.04, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)


ax3 = ax2.twinx()
ax3.plot(
    time_ticks, data_vint['t'],
    color='k', 
    label='T'
)
plt.gca().spines['top'].set_visible(False)
ax3.set_ylabel('K')
ax3.set_yticks(np.linspace(-0.5, 0.5, 6))
ax3.set_ylim(-0.52, 0.52)

plt.text(2.3, -1, 'Day After')
plt.text(-1.7, -1, 'Day Before')

cbar = fig.colorbar(
    cr1, cax=cax, ax=[ax1, ax2], location="right",  label="K/day")
cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6, 8, 10])

ax1.set_title(f'Exp: {exp}, Heating: LRF, Bandpass Filter: No\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen), TOT (black, dashed)',
fontsize=12, loc='left')
plt.savefig(f'/home/b11209013/2024_Research/MPAS/Composite/raw_composite/Raw_comp_image/LRF_sourced/{exp}_raw_comp.png', dpi=300)
# %%
