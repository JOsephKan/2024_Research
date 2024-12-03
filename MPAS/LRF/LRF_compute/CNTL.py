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


# 2. Running mean
def running_mean(data: np.ndarray, window_size: int, axis: int) -> np.ndarray:

    kernel = np.ones(window_size) / window_size

    arr_conv = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=axis, arr=data
    )

    return arr_conv


# 3. Normal Equation
def NormEqu(
    tend: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:

    comp1 = tend @ state.T
    comp2 = np.linalg.inv(
        state @ state.T + np.diag(np.array([1e-3]).repeat(state.shape[0]))
    )

    return comp1 @ comp2


# %%
# Load data
case: str = "NSC"

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
# Running mean data
data_running = Parallel(n_jobs=16)(
    delayed(running_mean)(data[var], 10, 3) for var in var_list
)


data_running: dict[str, np.ndarray] = {
    var: data_running[i].mean(axis=2) for i, var in enumerate(var_list)
}

print(f"shape of qv after running: ", data_running["qv"].shape)
# %%
# unit conversion
theta2t = (1000 / lev[None, :, None]) ** (-0.286)

data_convert: dict[str, np.ndarray] = {
    "t"        : data_running["theta"] * theta2t,
    "qv"       : data_running["qv"] * 1e3,
    "rtcuten"  : data_running["rthcuten"] * 86400 * theta2t,
    "rqvcuten" : data_running["rqvcuten"] * 1e3 * 86400,
    "rtratenlw": data_running["rthratenlw"] * 86400 * theta2t,
    "rtratensw": data_running["rthratensw"] * 86400 * theta2t,
}

var_list = data_convert.keys()

# Compute anomalies
data_ano: dict[str, np.ndarray] = {
    var: (data_convert[var] - data_convert[var].mean()).transpose(1, 0, 2).reshape(llev, -1)
    for var in var_list
}

# %% Compute linear response function
# 1. Compute state vector
state_vec: np.ndarray = np.concatenate((data_ano["t"], data_ano["qv"]), axis=0)
conv_tend: np.ndarray = np.concatenate((data_ano["rtcuten"], data_ano["rqvcuten"]), axis=0)

# 2. Compute LRF
lw_lrf: np.ndarray = NormEqu(np.array(data_ano['rtratenlw']), np.array(state_vec))
sw_lrf: np.ndarray = NormEqu(np.array(data_ano['rtratensw']), np.array(state_vec))
cu_lrf: np.ndarray = NormEqu(np.array(conv_tend), np.array(state_vec))

lw_lrf[:, 38:41] = np.nan
sw_lrf[:, 38:41] = np.nan
cu_lrf[:, 38:41] = np.nan

lrf_dict: dict[str, np.ndarray] = {
    'lw': lw_lrf,
    'sw': sw_lrf,
    'cu': cu_lrf
}

half = int(llev)

# %%
jl.dump(lrf_dict, f"/home/b11209013/2024_Research/MPAS/LRF/LRF_compute/LRF_file/{case}_lrf.joblib", compress=3)
