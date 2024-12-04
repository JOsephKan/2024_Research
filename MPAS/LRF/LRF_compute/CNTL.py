# This program is to compute LRF for CNTL
# %%
# import package
from os import path
import time
import numpy as np
import joblib as jl
import netCDF4 as nc

from joblib import Parallel, delayed
from matplotlib import pyplot as plt


# %%
# define function
# 1. Load data
def load_data(path: str, var: str, lat_lim) -> np.ndarray:

    with nc.Dataset(f"{path}{var}.nc", "r", mmap=True) as f:
        data = f.variables[var][:, :, lat_lim, :]

    return data

# 2. Normal Equation
def NormEqu(
    tend: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:

    comp1 = tend @ state.T
    comp2 = np.linalg.inv(state @ state.T)

    return comp1 @ comp2


# %%
# Load data
case: str = "CNTL"

path: str = f"/work/b11209013/2024_Research/MPAS/merged_data/{case}/"

var_list: list[str] = [
    "qv",
    "theta",
    "rqvcuten",
    "rthcuten",
    "rthratenlw",
    "rthratensw",
]

# load dimension information
with nc.Dataset(f"{path}qv.nc", "r", mmap=True) as f:
    lat: np.ndarray = f.variables["lat"][:]
    lon: np.ndarray = f.variables["lon"][:]
    lev: np.ndarray = f.variables["lev"][:]
    time: np.ndarray = f.variables["time"][:]

lat_lim = np.where((lat >= -5) & (lat <= 5))[0]

lat = lat[lat_lim]

# load variables
data = Parallel(n_jobs=16)(delayed(load_data)(path, var, lat_lim) for var in var_list)

data: dict[str, np.ndarray] = {var_list[i]: data[i] for i in range(len(var_list))}

ltime, llev, llat, llon = data["qv"].shape

# %%
# unit conversion
theta2t = (1000 / lev[None, :, None, None]) ** (-287.5/ 1004.5)

data_convert: dict[str, np.ndarray] = {
    "t"        : data["theta"] * theta2t - 273.15,
    "qv"       : data["qv"] * 1e3,
    "rtcuten"  : data["rthcuten"] * 86400 * theta2t,
    "rqvcuten" : data["rqvcuten"] * 1e3 * 86400,
    "rtratenlw": data["rthratenlw"] * 86400 * theta2t,
    "rtratensw": data["rthratensw"] * 86400 * theta2t,
}
var_list = data_convert.keys()

# Compute anomalies

data_convert['t']  -= data_convert['t'].mean(axis=(2, 3))[:, :, None, None]
data_convert['qv'] -= data_convert['qv'].mean(axis=(2, 3))[:, :, None, None]

data_rs: dict[str, np.ndarray] = {
    var: data_convert[var].transpose((1, 0, 2, 3)).reshape(llev, ltime * llat * llon)
    for var in var_list
}

# %% Compute linear response function
# 1. Compute state vector
state_vec: np.ndarray = np.concatenate((data_rs["t"], data_rs["qv"]), axis=0)
conv_tend: np.ndarray = np.concatenate((data_rs["rtcuten"], data_rs["rqvcuten"]), axis=0)

# 2. Compute LRF
lw_lrf: np.ndarray = NormEqu(np.array(data_rs['rtratenlw']), np.array(state_vec))
sw_lrf: np.ndarray = NormEqu(np.array(data_rs['rtratensw']), np.array(state_vec))
cu_lrf: np.ndarray = NormEqu(np.array(conv_tend), np.array(state_vec))

lrf_dict: dict[str, np.ndarray] = {
    'lw': lw_lrf,
    'sw': sw_lrf,
    'cu': cu_lrf
}

half = int(llev)
# %%

recon_lw = np.array(lw_lrf) @ np.array(state_vec)

plt.contourf(recon_lw[:, :100] - data_rs['rtratenlw'][:, :100], levels=100)
plt.colorbar()
# %%
jl.dump(lrf_dict, f"/home/b11209013/2024_Research/MPAS/LRF/LRF_compute/LRF_file/{case}_lrf.joblib", compress=3)
