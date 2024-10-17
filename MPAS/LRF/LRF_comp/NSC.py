# This program is to form the linear response function
# import package
import numpy as np
import netCDF4 as nc
import pickle as pkl
from matplotlib import pyplot as plt
import joblib
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

# ================== # 
# functions
# load data
def load_data(path, var, lim):
    with nc.Dataset(f"{path}{var}.nc", "r", mmap=True) as data:
        data = data[var][:, :, lim, :]
    return data

def running_mean(data, window_size, axis=3):
    kernel = np.ones(window_size) / window_size
    arr_convolve = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=axis, arr=data)

    return np.mean(arr_convolve, axis=2)

def interpolate(data, lev_org, lev_itp):
    data_itp = interp1d(lev_org[::-1], data[::-1], axis=0)(lev_itp)

    return data_itp

def NormEqu(tend, state):
    comp1 = tend @ state.T
    comp2 = np.linalg.inv(state @ state.T + np.diag(np.array([0.1]).repeat(state.shape[0])))

    return comp1 @ comp2

# ================== #
# load data
case = "NSC"
path = f"/work/b11209013/MPAS/tropic_data/{case}/"
var_list = [
    "qv", "theta", "rqvcuten", "rthcuten",
    "rthratenlw", "rthratensw"
    ]

## load dimension
with nc.Dataset(f"{path}qv.nc", "r", mmap=True) as data:
    lon  = data["lon"][:]
    lat  = data["lat"][:]
    lev  = data["lev"][:]
    time = data["time"][:]

lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
lat = lat[lat_lim]

## load data
data = Parallel(n_jobs=12)(
    delayed(load_data)(path, var, lat_lim)
    for var in var_list
    )

data = {var: data[i] for i, var in enumerate(var_list)}

ltime, llev, llat, llon = data["qv"].shape

# ================== #
# data running mean
data_running = Parallel(n_jobs=12)(
    delayed(running_mean)(data[var], 10)
    for var in var_list
    )

data_running = {var: data_running[i] for i, var in enumerate(var_list)}

del data

# ================== #
# unit conversion
data_convert = {}
theta2t = (1000/lev[None, :, None])**(-0.286)

data_convert["t"]         = data_running["theta"]*theta2t
data_convert["qv"]        = data_running["qv"]*1000
data_convert["rqvcuten"]  = data_running["rqvcuten"]*1000*86400
data_convert["rtcuten"]   = data_running["rthcuten"]*86400*theta2t
data_convert["rtratenlw"] = data_running["rthratenlw"]*86400*theta2t
data_convert["rtratensw"] = data_running["rthratensw"]*86400*theta2t

del data_running

var_list = data_convert.keys()

# ================== # 
# compute anomaly 
data_ano = {var: data_convert[var] - data_convert[var].mean() for var in var_list}

del data_convert

# ================== #
# permute and interpolate
## permute and reshape
data_ano = {var: data_ano[var].transpose(1, 0, 2).reshape((llev, -1)) for var in var_list}

## interpolate
lev_itp = np.linspace(100, 1000, 19)

data_itp = Parallel(n_jobs=12)(
    delayed(interpolate)(data_ano[var], lev, lev_itp)
    for var in var_list
    )

data_itp = {var: data_itp[i] for i, var in enumerate(var_list)}

# ================ #
# form state vector and tendency
state_vec = np.concatenate((data_itp["t"], data_itp["qv"]), axis=0)
conv_tend = np.concatenate((data_itp["rtcuten"], data_itp["rqvcuten"]), axis=0)

## linear response function
lw_lrf = NormEqu(data_itp["rtratenlw"], state_vec)
sw_lrf = NormEqu(data_itp["rtratensw"], state_vec)
cu_lrf = NormEqu(conv_tend, state_vec)

lw_lrf[:, 19:23] = np.nan
sw_lrf[:, 19:23] = np.nan
cu_lrf[:, 19:23] = np.nan

lrf_dict = {
    "lev"   : lev_itp,
    "lw_lrf": lw_lrf,
    "sw_lrf": sw_lrf,
    "cu_lrf": cu_lrf
}

# ================= #
# save as pkl
joblib.dump(lrf_dict, f"/home/b11209013/2024_Research/MPAS/LRF/LRF_new/LRF_file/lrf_{case}.pkl")

# ================= #
# plot out linear response function
plt.rcParams.update({
    'font.size': 12,
    'figure.titlesize': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'font.family': 'serif',
})

image_path = f"/home/b11209013/2024_Research/MPAS/LRF/LRF_new/image/{case}/"

## longwave effect
### T - T
plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, lw_lrf[:, :19], cmap="RdBu_r", vmin=-1.5, vmax=1.5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r" $\frac{dT}{dt})_{LW}$ from Temperature")
plt.colorbar(orientation='horizontal', label='[(K/day) / (K) ]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}lw_t_t.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, lw_lrf[:, 19:], cmap="RdBu_r", vmin=-1.5, vmax=1.5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dT}{dt})_{LW}$ from Moisture')
plt.colorbar(orientation='horizontal', label='[(K/day) / (g/kg)]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}lw_t_q.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, sw_lrf[:, :19], cmap="RdBu_r", vmin=-3, vmax=3)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dT}{dt})_{SW}$ from Temperature')
plt.colorbar(orientation='horizontal', label='[(K/day) / (K) ]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}sw_t_t.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, sw_lrf[:, 19:], cmap="RdBu_r", vmin=-3, vmax=3)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dT}{dt})_{SW}$ from Moisture')
plt.colorbar(orientation='horizontal', label='[(K/day) / (g/kg)]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}sw_t_q.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, cu_lrf[:19, :19], cmap="RdBu_r", vmin=-5, vmax=5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dT}{dt})_{CU}$ from Temperature')
plt.colorbar(orientation='horizontal', label='[(K/day) / (K) ]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}cu_t_t.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, cu_lrf[:19, 19:], cmap="RdBu_r", vmin=-5, vmax=5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dT}{dt}_{CU}$ from Moisture')
plt.colorbar(orientation='horizontal', label='[(K/day) / (g/kg)]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}cu_t_q.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, cu_lrf[19:, :19], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dq}{dt}_{CU}$ from Temperature')
plt.colorbar(orientation='horizontal', label='[(K/day) / (K) ]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}cu_q_t.png', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(8, 7))
plt.pcolormesh(lev_itp, lev_itp, cu_lrf[19:, 19:], cmap="RdBu_r", vmin=-5, vmax=5)
plt.xticks(np.linspace(100, 1000, 10))
plt.yticks(np.linspace(100, 1000, 10))
plt.xlim(1000, 100)
plt.ylim(1000, 100)
plt.xlabel('State Vecter Level (hPa)')
plt.ylabel('Tendency Level (hPa)')
plt.title(case+r' $\frac{dq}{dt})_{CU}$ from Moisture')
plt.colorbar(orientation='horizontal', label='[(K/day) / (g/kg)]', aspect=50, shrink=0.8)
plt.savefig(f'{image_path}cu_q_q.png', dpi=500)
plt.show()
plt.close()

