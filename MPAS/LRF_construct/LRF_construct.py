# This program is to construct linear response function for MPAS experiments
# %% section 1: import package
import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# %% section 2: define functions
## 2.1: read data
def load_data(
    path : str,
    var  : str,
    lim  : list,
    ) -> np.ndarray:
    
    with nc.Dataset(f'{path}{var}.nc', 'r') as f:
        data: np.ndarray = f.variables[var][:, :, lim, :]
        
    return var, data

# %% section 3: load data
## data parameters
#exp : str = 'CNTL' # experiment
exp : str = sys.argv[1] # experiment

path: str = f'/work/b11209013/2024_Research/MPAS/merged_data/{exp}/' # path to data

var_list: list[str] = [
    'qv', 'theta', 'rthcuten', 'rthratenlw', 'rthratensw'
]

## load coordinateas
dims: dict[str, np.ndarray] = dict()

with nc.Dataset(f'{path}{var_list[0]}.nc', 'r') as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][:]
    

lat_lim: list[int] = np.where(
    (dims['lat'] >= -5) & (dims['lat'] <= 5)
    )[0]
    
## load variables
data: dict[str, np.ndarray] = dict(
    Parallel(n_jobs = -1)(
        delayed(load_data)(path, var, lat_lim)
        for var in var_list
    )
)

ltime, llev, llat, llon = data[var_list[0]].shape    

# %% section 4: Processing data
## 4.1 convert unit
theta2t = lambda theta: theta*(1000/dims['lev'][None, :, None, None])**-0.286

data_convert: dict[str, np.ndarray] = {
    't' : theta2t(data['theta']),
    'cu': theta2t(data['rthcuten'])*86400,
    'lw': theta2t(data['rthratenlw'])*86400,
    'sw': theta2t(data['rthratensw'])*86400,
    'qv': data['qv']*1000,
}

## 4.2 comopute anomaly
data_ano: dict[str, np.ndarray] = {
    key: data_convert[key] - data_convert[key].mean(axis = (0, 2, 3))[None, :, None, None]
    for key in data_convert.keys()
}

## 4.3 permute and reshape data
data_rs: dict[str, np.ndarray] = {
    key: data_ano[key].transpose(1, 0, 2, 3).reshape(llev, -1)
    for key in data_ano.keys()
}

data_rs['tot'] = data_rs['lw'] + data_rs['sw'] + data_rs['cu']

# %% section 5: Construct LRF
## 5.1 define normal equation
normal_equation = lambda x, y: x @ y.T @ np.linalg.inv(y @ y.T) # x: tend; y: state vector

## 5.2 constructe state vector
state_vec: np.ndarray = np.concatenate(
    [data_rs['t'], data_rs['qv']], axis=0
)

lrf: dict[str, np.ndarray] = {
    'lw' : normal_equation(np.array(data_rs['lw'] ), np.array(state_vec)),
    'sw' : normal_equation(np.array(data_rs['sw'] ), np.array(state_vec)),
    'cu' : normal_equation(np.array(data_rs['cu'] ), np.array(state_vec)),
    'tot': normal_equation(np.array(data_rs['tot']), np.array(state_vec))
}
# %% section 6: constrain data to 200hPa
lev_lim = np.argmin(np.abs(dims['lev'] - 200))

for var in lrf.keys():
    lrf[var][:, lev_lim+1:] = np.nan

# %% section 7: save data
save_path = f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/'

with nc.Dataset(f"{save_path}{exp}.nc", "w") as f:
    f.createDimension("lev_state", llev*2)  
    f.createDimension("lev_tend", llev)
  
    lev_var = f.createVariable("lev", np.float64, ("lev_tend"))
    lev_var.description="Levels of EOF modes"
    lev_var[:] = dims['lev']

    lw_var = f.createVariable("lw", np.float64, ("lev_tend", "lev_state"))
    lw_var.description="LRF for longwave heating"
    lw_var[:] = lrf['lw']

    sw_var = f.createVariable("sw", np.float64, ("lev_tend", "lev_state"))
    sw_var.description="LRF for shortwave heating"
    sw_var[:] = lrf['sw']

    cu_var = f.createVariable("cu", np.float64, ("lev_tend", "lev_state"))
    cu_var.description="LRF for cumulus heating"
    cu_var[:] = lrf['cu']

    tot_var = f.createVariable("tot", np.float64, ("lev_tend", "lev_state"))
    tot_var.description="LRF of total heating"
    tot_var[:] = lrf['tot']
