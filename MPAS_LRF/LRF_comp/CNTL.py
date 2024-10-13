# Thie program is to compute linear response function of MPAS CNTL
# import pacakges
import dask.array as da
import numpy as np
import pickle as pkl
import netCDF4 as nc
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from multiprocessing import Pool
import sys
sys.setrecursionlimit(2000)
# ============ #
# functions
# load data
def load_data(var):
    with nc.Dataset(f"/work/b11209013/MPAS/tropic_data/CNTL/{var}.nc", "r", mmap=True) as f:
        data = da.from_array(f[var][:], chunks="auto")
    return data

# regularized normal equation
def normal_equation(tend, state):
    comp1 = tend @ state.T
    comp2 = np.linalg.inv(state @ state.T + np.diag(np.random.rand(state.shape[0])*1e-1))

    return comp1 @ comp2

# ============ #
# load data
path = "/work/b11209013/MPAS/tropic_data/CNTL/"

var_list = ["theta", "qv", "rthcuten", "rqvcuten", "rthratenlw", "rthratensw"]

## load dimension
with nc.Dataset(path+"theta.nc", "r", mmap=True) as f:
    lat = f["lat"][:]
    lon = f["lon"][:]
    lev = f["lev"][:]
    time = f["time"][:]

## load data
data = {v: load_data(v) for v in var_list}

# ============= #
# unit transformation
data_trans = {}

trans_factor = (1000/lev[None, :, None, None])**(-0.286)

data_trans["qv"]        = data["qv"]*1000 
data_trans["rqvcuten"]  = data["rqvcuten"]*1000
data_trans["t"]         = data["theta"]*trans_factor
data_trans["rtcuten"]   = data["rthcuten"]*trans_factor*86400
data_trans["rtratenlw"] = data["rthratenlw"]*trans_factor*86400
data_trans["rtratensw"] = data["rthratensw"]*trans_factor*86400

del data

# remove time mean
data_trans = {v: data_trans[v] - data_trans[v].mean(axis=0) for v in data_trans.keys()}

# ============= #
# interpolate data into specific level
lev_itp = np.linspace(100, 1000, 10)
data_itp = {v: interp.interp1d(lev[::-1], data_trans[v][:, ::-1], axis=1)(lev_itp) for v in data_trans.keys()}

del data_trans
# ============= #
# chunking data 
## zonal chunking, meridional average
## setup conditions
left  = np.linspace(0.25, 357.75, 716)
right = np.linspace(2.25, 359.75, 716)

cond = [np.where((lon >= l) & (lon <= r)) for l, r in zip(left, right)]

## chunking
data_chunk = {}

for v in data_itp.keys():
    var_trans = data_itp[v].transpose((1, 0, 2, 3))

    var_temp = da.empty((len(lev_itp), len(time), len(cond)))
    for i, c in enumerate(cond):
        var_temp[:, :, i] = np.squeeze(var_trans[:, :, :, c]).mean(axis=(2, 3)).reshape((len(lev_itp), len(time)))

    var_temp = var_temp.reshape((len(lev_itp), len(time)*len(cond)))

    data_chunk[v] = var_temp

# ============= #
# concatenate into state vector and tendencies
state_vec = da.concatenate((data_chunk["t"], data_chunk["qv"]), axis=0).compute()

lw_effect = data_chunk["rtratenlw"].compute()
sw_effect = data_chunk["rtratensw"].compute()
cu_effect = da.concatenate((data_chunk["rtcuten"], data_chunk["rqvcuten"]), axis=0).compute()

# ============ #
# construct linear response function
lw_lrf = normal_equation(np.array(lw_effect), np.array(state_vec))
sw_lrf = normal_equation(np.array(sw_effect), np.array(state_vec))
cu_lrf = normal_equation(np.array(cu_effect), np.array(state_vec))

lrf = {
    "lw_t_t" : lw_lrf[:, :10]  , "lw_t_q" : lw_lrf[:, 10:],
    "sw_t_t" : sw_lrf[:, :10]  , "sw_t_q" : sw_lrf[:, 10:],
    "cu_t_t" : cu_lrf[:10, :10], "cu_t_q" : cu_lrf[:10, 10:],
    "cu_q_t" : cu_lrf[10:, :10], "cu_q_q" : cu_lrf[10:, 10:]
}

with open("/work/b11209013/MPAS/LRF/CNTL.pkl", "wb") as f:
    pkl.dump(lrf, f)
