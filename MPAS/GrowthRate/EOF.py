# This program is to compute Q1 EOF structure in MPAS CNTL experiment
# import packages
import sys
import numpy as np

from netCDF4 import Dataset
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

sys.path.append("/home/b11209013/Package/")
import SignalProcess as sp

# ==================== # 
# load data
fname: str = '/work/b11209013/MPAS/merged_data/CNTL/q1.nc'

with Dataset(fname, 'r') as f:
    lat  = f.variables['lat'][:]
    lon  = f.variables['lon'][:]
    lev  = f.variables['lev'][:]
    time = f.variables['time'][:]
    
    lat_lim = np.where((lat>=-5) & (lat<=5))[0]
    q1 = f.variables['q1'][:, :, lat_lim, :]*86400/1004.5

ltime, llev, llat, llon = q1.shape

# ====================== #
# permute and reshape
q1_pm = q1.transpose((1, 0, 2, 3))
q1_rs = q1_pm.reshape((llev, -1))

# ====================== # 
# interpolate
lev_itp = np.linspace(150, 1000, 18)

q1_itp = interp1d(lev[::-1], q1_rs[::-1], axis=0)(lev_itp)

q1_itp_ano = q1_itp - q1_itp.mean()

# ====================== #
# compute EOF of q1
q1_e = sp.EOF(q1_itp_ano)

expvar, q1_eof, q1_pc = q1_e.EmpOrthFunc()

# ====================== #
# save EOF as txt file
np.savetxt('PCA_file/CNTL_EOF.txt', q1_eof)
