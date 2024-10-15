# This program is to compute linear response function from CloudSat
# import package
import numpy as np
import pickle as pkl
import netCDF4 as nc
from joblib import load, Parallel, delayed

# ================ #
# functions
# chunking data
def chunking(lat, lon, qlw, qsw, cond):
    left, right, top, bot = cond

    bound = np.where(
        (lat >= bot) & (lat <= top) & (lon >= left) & (lon <= right)
    )[0]

    qlw_new = qlw[:, bound]
    qsw_new = qsw[:, bound]

    return qlw_new, qsw_new

# load data
# ERA5

## cloudsat
data_cs = load("/work/b11209013/LRF/data/CloudSat/CloudSat.pkl", mmap_mode="r")

# chunking data
bot = np.linspace(-5, 3, 17)
top = np.linspace(-3, 5, 17)
left = np.linspace(160, 258, 195)
right = np.linspace(162, 260, 195)

cond = [(left[j], right[j], top[i], bot[i]) for i in range(len(top)) for j in range(len(left))]


