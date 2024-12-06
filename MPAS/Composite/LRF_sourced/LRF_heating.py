# This program is to compute the compositing of CNTL
# %% section 1
# import package
import os
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append('/home/b11209013/Package/')
import Theory as th # type: ignore
import DataProcess as dp # type: ignore
import SignalProcess as sp # type: ignore

# %%
# Functions
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


# %% section 2
# load data
# case name
case = sys.argv[1]

# variable list
var_list = ['t', 'qv', 'q1']

# path
fname = f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.joblib'

# load principal component data
data = jl.load(fname)

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
lrf = jl.load(f'/home/b11209013/2024_Research/MPAS/LRF/LRF_compute/LRF_file/lrf_{case}.joblib')

lw  = np.where(np.isnan(lrf['lw_lrf'])==True, 0, lrf['lw_lrf'])
sw  = np.where(np.isnan(lrf['sw_lrf'])==True, 0, lrf['sw_lrf'])
cu  = np.where(np.isnan(lrf['cu_lrf'])==True, 0, lrf['cu_lrf'])
tot = np.where(np.isnan(lrf['tot_lrf'])==True, 0, lrf['tot_lrf'])

# load reference longitude and time for compositing
ref = np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/{case}.npy')
lon_ref = ref[0]
time_ref = ref[1]

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

# %%
# select data
time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(len(time_ref))
]

time_ticks = np.linspace(-4, 4, 33)

# data selection
data_sel: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: np.array([
            reconstruct[pc][var][time_itv[i], lon_ref[i]]
            for i in range(len(time_ref))
        ]).mean(axis=0)
        for var in reconstruct[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

plt.plot(time_ticks, data_sel['pc1']['t'])
plt.plot(time_ticks, data_sel['pc2']['t'])

# %% 
# Generate heating profile from LRF
# 1. construct vertical profile of t and qv
vert_prof: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: np.matrix(eof1).T @ np.matrix(data_sel['pc1'][var])
        for var in ['t', 'qv', 'q1']
    },
    'pc2': {
        var: np.matrix(eof2).T @ np.matrix(data_sel['pc2'][var])
        for var in ['t', 'qv', 'q1']
    }
}

plt.contourf(time_ticks, lev, vert_prof['pc1']['t'] + vert_prof['pc2']['t'])
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.colorbar()

# 2. construct state vector
state_vec: dict[str, np.ndarray] = {
    'pc1': np.concatenate([vert_prof['pc1']['t'], vert_prof['pc1']['qv']], axis=0),
    'pc2': np.concatenate([vert_prof['pc2']['t'], vert_prof['pc2']['qv']], axis=0)
}

# 3. compute heating profile
heating: dict[str, dict[str, np.ndarray]] = {
    pc: {
        'lw': lw @ state_vec[pc],
        'sw': sw @ state_vec[pc],
        'cu': cu @ state_vec[pc],
        'tot': tot @ state_vec[pc]
    }
    for pc in ['pc1', 'pc2']
}

heating['vert_prof'] = vert_prof
heating['lev'] = lev
heating['time_tick'] = time_ticks

# %%
# save file
jl.dump(heating, f'/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/LRF_com_heating/{case}_heating.joblib')