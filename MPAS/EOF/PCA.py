# This program is to compute EOF of Q1 in MPAS experiments
# The range for EOF is set to be -5 to 5 degree
# import package
import os
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

## import sself-defined package
sys.path.append('/home/b11209013/Package')
import SignalProcess as sp  #type: ignore

## imoprt system parameters
case = sys.argv[1]
def main():
# ===================== Part 1: load data ==================== #
    # path for case data
    path: str = f'/work/b11209013/2024_Research/MPAS/merged_data/{case}/'
    
    data: dict[str, np.ndarray] = {}
    
    # load Q1 dataset
    with nc.Dataset(f'{path}q1.nc', 'r') as f:
        lon : np.ndarray = f.variables['lon'][:]
        lat : np.ndarray = f.variables['lat'][:]
        lev : np.ndarray = f.variables['lev'][:]
        time: np.ndarray = f.variables['time'][:]
        
        lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
        
        lat : np.ndarray = lat[lat_lim]
        data['q1']       = f.variables['q1'][:, :, lat_lim, :] *86400/1004.5
    
    # define functions for converting theta to temperature
    def theta2temp(theta: np.ndarray) -> np.ndarray:
        return theta * (1000 / lev[None, :, None, None])**(-0.286)

    # load potential temperature dataset
    with nc.Dataset(f'{path}theta.nc', 'r') as f:
        data['t']    = theta2temp(f.variables['theta'][:, :, lat_lim, :])
    
    # load moisture dataset
    with nc.Dataset(f'{path}qv.nc', 'r') as f:
        data['qv']       = f.variables['qv'][:, :, lat_lim, :]*1000
        
    # load LW heating dataset
    with nc.Dataset(f'{path}rthratenlw.nc', 'r') as f:
        data['lw']       = theta2temp(f.variables['rthratenlw'][:, :, lat_lim, :])*86400
    
    # load SW heating dataset
    with nc.Dataset(f'{path}rthratensw.nc', 'r') as f:
        data['sw']       = theta2temp(f.variables['rthratensw'][:, :, lat_lim, :])*86400
        
    # load Cu heating dataset
    with nc.Dataset(f'{path}rthcuten.nc', 'r') as f:
        data['cu']       = theta2temp(f.variables['rthcuten'][:, :, lat_lim, :])*86400
    
    # load EOF dataset
    EOF   : np.ndarray = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/EOF.joblib')

    eof = EOF['EOF']
    print(eof.shape)

    # ===================== Part 1.5: reshape and itp dataset ====== #
    data_rs: dict[str, np.ndarray] = {
        var: np.reshape(data[var].transpose((1, 0, 2, 3)), (lev.size, -1))
        for var in data.keys()
    }

    lev_itp = np.linspace(150, 1000, 18)

    data_itp: dict[str, np.ndarray] = {
        var: interp1d(lev[::-1], data_rs[var][::-1], axis=0)(lev_itp)
        for var in data_rs.keys()
    }

    # ===================== Part 2: PCA analysis ==================== #
    # project the data onto the EOF1 and EOF2

    data_ano: dict[str, np.ndarray] = {
        var: data_rs[var] - np.mean(data_rs[var])
        for var in data_rs.keys()
    }

    # PC dataset
    pc_comp = lambda eof, data: np.linalg.inv(eof.T @ eof) @ eof.T @ data
    
    data_pc: dict[str, np.ndarray] = {
        var: pc_comp(np.array(eof), np.array(data_ano[var]))
        for var in data_ano.keys()
    }

    data_pc = {
        var: data_pc[var].reshape((38, time.size, lat.size, lon.size))
        for var in data_pc.keys()
    }
    
    ## output file
    output_dict = {
        'lon' : lon,
        'lat' : lat,
        'time': time,
        'pc': data_pc,
    }
    
    # ===================== Part 3: save data ==================== #
    # path for saving data
    path_save: str = f'/work/b11209013/2024_Research/MPAS/PC/'
    jl.dump(output_dict, f'{path_save}{case}_PC.joblib', compress=('zlib', 1))

# ===================== Execution ==================== #
if __name__ == '__main__':
    main()