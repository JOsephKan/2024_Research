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


# %% section 2
# load data
# case name
case = sys.argv[1]
#case = 'CNTL'
# variable list
var_list = ['t', 'qv', 'q1']

# path
fname = f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.nc'

# load principal component data
data = jl.load(fname)

lon  = data['lon']
lat  = data['lat']
time = data['time']
pc_data = data['pc']

data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: data['pc'][var][0].mean(axis=1)
        for var in var_list
    },
    'pc2': {
        var: data['pc'][var][1].mean(axis=1)
        for var in var_list
    }
}

ltime, llon = data['pc1']['t'].shape
plt.contourf(lon, time, data['pc1']['t'])
plt.colorbar()

# size of the pc: (time, lon)

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

# load reference longitude and time for compositing
ref = np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/{case}.npy')
lon_ref = ref[0]
time_ref = ref[1]


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
            data[pc][var][time_itv[i], lon_ref[i]]
            for i in range(len(time_ref))
        ]).mean(axis=0)
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %% 
# Generate heating profile from LRF
# 1. construct vertical profile of t and qv
vert_prof: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: np.matrix(eof1).T @ np.matrix(data_sel['pc1'][var] - data_sel['pc1'][var].mean())
        for var in ['t', 'qv', 'q1']
    },
    'pc2': {
        var: np.matrix(eof2).T @ np.matrix(data_sel['pc2'][var] - data_sel['pc2'][var].mean())
        for var in ['t', 'qv', 'q1']
    }
}

plt.contourf(time_ticks, lev, vert_prof['pc1']['t']+vert_prof['pc2']['t'])
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
        'cu': cu @ state_vec[pc]
    }
    for pc in ['pc1', 'pc2']
}

heating['vert_prof'] = vert_prof
heating['lev'] = lev
heating['time_tick'] = time_ticks

# %%
# save file
jl.dump(heating, f'/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/LRF_com_heating/{case}_heating_no_filt.joblib')
