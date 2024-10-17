# This program is to apply PCA on other experiments with CNTL Q1 EOF as its Basis
# import packages
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 

# ==================== #
# functions
# normal equation
def normal_equation(eof: np.ndarray, data:np.ndarray) -> np.ndarray:
    comp1 = eof.T @ data
    comp2 = np.linalg.inv(eof.T @ eof)
    pc = comp2 @ comp1

    return np.array(pc)

# ===================== #
# load data
case = 'CNTL'

with nc.Dataset(f'/work/b11209013/MPAS/merged_data/{case}/theta.nc', 'r') as f:
    lon  = f.variables['lon'][:]
    lat  = f.variables['lat'][:]
    lev  = f.variables['lev'][:]
    time = f.variables['time'][:]
    
    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]

    lat = lat[lat_lim]
    theta = f.variables['theta'][:, :, lat_lim, :]

ltime, llev, llat, llon = theta.shape

with nc.Dataset(f'/work/b11209013/MPAS/merged_data/{case}/q1.nc', 'r') as f:
    q1 = f.variables['q1'][:, :, lat_lim, :]*86400/1004.5

q1_eof = np.loadtxt('/home/b11209013/2024_Research/MPAS/LRF/LRF_new/PCA_file/CNTL_EOF.txt')

eof1 = np.matrix(q1_eof[:, 0]).T
eof2 = np.matrix(q1_eof[:, 1]).T

# ====================== #
# convert theta into temperature
t = theta * (1000/lev[None, :, None, None])**(-0.286)

# ====================== #
# permute and reshape
t_pm = t.transpose((1, 0, 2, 3))
q1_pm = q1.transpose((1, 0, 2, 3))

t_rs = t_pm.reshape((llev, -1))
q1_rs = q1_pm.reshape((llev, -1))

# ====================== #
# interpolate
lev_itp = np.linspace(100, 1000, 19)

t_itp = interp1d(lev[::-1], t_rs[::-1], kind="linear", axis=0)(lev_itp)
q1_itp = interp1d(lev[::-1], q1_rs[::-1], kind="linear", axis=0)(lev_itp)
# ====================== #
# compute PC of q1 and EOF
t_itp = t_itp - t_itp.mean()
q1_itp = q1_itp - q1_itp.mean()

tpc1  = normal_equation(eof1, t_itp)
tpc2  = normal_equation(eof2, t_itp)
q1pc1 = normal_equation(eof1, q1_itp)
q1pc2 = normal_equation(eof2, q1_itp)

# ====================== # 
# reshape PC
tpc1  = tpc1.reshape((ltime, llat, llon))
tpc2  = tpc2.reshape((ltime, llat, llon))
q1pc1 = q1pc1.reshape((ltime, llat, llon))
q1pc2 = q1pc2.reshape((ltime, llat, llon))

# ======================= #
# save nc file
with nc.Dataset(f'/home/b11209013/2024_Research/MPAS/LRF/LRF_new/PCA_file/{case}_PC.nc', 'w') as f:
    f.createDimension('lon', llon)
    f.createDimension('lat', llat)
    f.createDimension('time', ltime)

    lat_dim = f.createVariable('lat', 'f4', 'lat')
    lat_dim[:] = lat
    lat_dim.units = 'degree_north'

    lon_dim = f.createVariable('lon', 'f4', 'lon')
    lon_dim[:] = lon
    lon_dim.units = 'degree_east'

    time_dim = f.createVariable('time', 'f4', 'time')
    time_dim[:] = time
    time_dim.units = 'hours since 1900-01-01 00:00:00.0'

    tpc1_var = f.createVariable('tpc1', 'f4', ('time', 'lat', 'lon'))
    tpc1_var[:] = tpc1
    tpc1_var.units = 'K'

    tpc2_var = f.createVariable('tpc2', 'f4', ('time', 'lat', 'lon'))
    tpc2_var[:] = tpc2
    tpc2_var.units = 'K'

    q1pc1_var = f.createVariable('q1pc1', 'f4', ('time', 'lat', 'lon'))
    q1pc1_var[:] = q1pc1
    q1pc1_var.units = 'K/day'

    q1pc2_var = f.createVariable('q1pc2', 'f4', ('time', 'lat', 'lon'))
    q1pc2_var[:] = q1pc2
    q1pc2_var.units = 'K/day'
