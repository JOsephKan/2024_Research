# This program is to composite the raw data
# %% section 1: environment setting
## import package
from math import exp
import os
from resource import struct_rusage
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc
from matplotlib import pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from panel import state

## load self-defined package
sys.path.append('/home/b11209013/Package')
import DataProcess as dp         #type: ignore

## import system environment parameters
#exp: str = sys.argv[1] # load experiment name
exp: str = 'CNTL'

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
# 1. FFT2
def fft2(
        data: np.ndarray,
) -> np.ndarray:
    fft = np.array([np.fft.fft(data[i]) for i in range(data.shape[0])])
    fft = np.array([np.fft.ifft(fft[:, i]) for i in range(fft.shape[1])]).T
    return fft

# 2. IFFT2
def ifft2(
        data: np.ndarray,
) -> np.ndarray:
    ifft = np.array([np.fft.fft(data[:, i]) for i in range(data.shape[1])]).T
    ifft = np.array([np.fft.ifft(ifft[i]) for i in range(ifft.shape[0])])
    return ifft.real

# 3. Mask
def mask(wnm, frm):
    """
    Checks if (wnm, frm) lies in either the positive or negative region.

    Parameters:
        wnm: The wavenumber.
        frm: The frequency.

    Returns:
        A tuple of booleans (is_in_positive_region, is_in_negative_region):
        - is_in_positive_region: True if (wnm, frm) lies in the positive region.
        - is_in_negative_region: True if (wnm, frm) lies in the negative region.
    """
    kel_curves = lambda ed, k: (86400/(2*np.pi*6.371e6))*np.sqrt(9.81*ed)*k

    is_in_positive_region = (
        (wnm >= 1) & (wnm <= 14) &
        (frm >= 1 / 20) & (frm <= 1 / 2.5) &
        (frm <= kel_curves(90, wnm)) &
        (frm >= kel_curves(8, wnm))
    )
    
    is_in_negative_region = (
        (wnm <= -1) & (wnm >= -14) &
        (frm <= -1 / 20) & (frm >= -1 / 2.5) &
        (frm >= kel_curves(90, wnm)) &
        (frm <= kel_curves(8, wnm))
    )
    
    return np.where(is_in_positive_region | is_in_negative_region, 1, 0)

# %% section 2: Load data
# %% section 2
# load data
# case name
#case = sys.argv[1]
case='CNTL'
# path
fname = f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.joblib'

# load principal component data
data = jl.load(fname)

var_list: list[str] = ['t', 'qv', 'q1']

lon  = data['lon']
lat  = data['lat']
time = data['time']
pc_data = data['pc']

data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: data['pc'][var][0]
        for var in var_list
    },
    'pc2': {
        var: data['pc'][var][1]
        for var in var_list
    }
}

ltime, llat, llon = data['pc1']['t'].shape

# size of the pc: (time, lat, lon)

# load EOF structure
eof = jl.load('/work/b11209013/2024_Research/MPAS/PC/EOF.joblib')

lev : np.ndarray = eof['lev']
eof1: np.ndarray = eof['EOF'][:, 0]
eof2: np.ndarray = eof['EOF'][:, 1]

# load LRF file
lrf = jl.load(f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/LRF_{case}.joblib')

lw  = np.where(np.isnan(lrf['lw'])==True , 0, lrf['lw'])
sw  = np.where(np.isnan(lrf['sw'])==True , 0, lrf['sw'])
cu  = np.where(np.isnan(lrf['cu'])==True , 0, lrf['cu'])
tot = np.where(np.isnan(lrf['tot'])==True, 0, lrf['tot'])

# load reference longitude and time for compositing
lon_ref, time_ref = list(
    np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/Q1_sel/{case}.npy')
    )
# %%
# format data
fmt = dp.Format(lat)

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fmt.sym(data[pc][var])
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %%
# FFT on the symmetric data
fft: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fft2(sym[pc][var])
        for var in sym[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %%
# Bandpass filter
wn = np.fft.fftfreq(llon, d=1/llon).astype(int)
fr = np.fft.fftfreq(ltime, d=1/4)

wnm, frm = np.meshgrid(wn, fr)
mask_arr = mask(wnm, frm)

reconstruct: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: ifft2(fft[pc][var] * mask_arr)
        for var in fft[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %% section 3: form heating
# 3.1 construct state vector
state_vector: dict[str, np.ndarray] = {
    'pc1': np.concatenate([
        eof1[:, None] @ reconstruct['pc1']['t'].flatten()[None, :],
        eof1[:, None] @ reconstruct['pc1']['qv'].flatten()[None, :]
    ], axis=0),
    'pc2': np.concatenate([
        eof2[:, None] @ reconstruct['pc2']['t'].flatten()[None, :],
        eof2[:, None] @ reconstruct['pc2']['qv'].flatten()[None,:]
    ], axis=0)
}

heating: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        'lw' : (lw @ state_vector['pc1'] ).reshape((len(lev), ltime, llon)),
        'sw' : (sw @ state_vector['pc1'] ).reshape((len(lev), ltime, llon)),
        'cu' : (cu @ state_vector['pc1'] ).reshape((len(lev), ltime, llon)),
        'tot': (tot @ state_vector['pc1']).reshape((len(lev), ltime, llon))
    },
    'pc2': {
        'lw' : (lw @ state_vector['pc2'] ).reshape((len(lev), ltime, llon)),
        'sw' : (sw @ state_vector['pc2'] ).reshape((len(lev), ltime, llon)),
        'cu' : (cu @ state_vector['pc2'] ).reshape((len(lev), ltime, llon)),
        'tot': (tot @ state_vector['pc2']).reshape((len(lev), ltime, llon))
    }
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
    pc:{
        np.array([
            heating[pc][key][lev, time_itv[i], lon_ref[i]]
            for i in range(time_ref.size)
        ])
        for key in heating[pc].keys()
    }
    for pc in heating.keys()
}
print(data_sel['pc1'].keys())
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
