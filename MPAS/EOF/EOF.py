# This program is to compute EOF of Q1 in MPAS experiments
# The range for EOF is set to be -5 to 5 degree
# %% import package
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

sys.path.append('/home/b11209013/Package')
from SignalProcess import EOF

## imoprt system parameters
case = 'CNTL'

# %% ================== Part 1: load data ==================== #
# path for case data
path: str = f'/work/b11209013/2024_Research/MPAS/merged_data/{case}/'

# load Q1 dataset
with nc.Dataset(f'{path}q1.nc', 'r') as f:
    lat : np.ndarray = f.variables['lat'][:]
    lon : np.ndarray = f.variables['lon'][:]
    lev : np.ndarray = f.variables['lev'][:]
    
    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
    
    q1 = f.variables['q1'][:, :, lat_lim, :]
    


# %% ================== Part 1.5 : permute and reshape ==================== #
# permute and reshape variables
q1_rs = np.reshape(q1.transpose((1, 0, 2, 3)), (lev.size, -1))

# %% =================== Part 2: interpolate ==================== #
#lev_itp = np.linspace(150, 1000, 18).astype(int)
#q1_itp = interp1d(lev[::-1], q1_rs[::-1], axis=0, fill_value='extrapolate')(lev_itp)
q1_itp = q1_rs

# %% ==================== Part 3: EOF =========================== #
# EOF analysis

q1_ano = (q1_itp - np.mean(q1_itp))

q1_eof = EOF(q1_ano)

exp, eof, pc = q1_eof.EmpOrthFunc()

plt.plot(eof[:, 0], lev)
plt.plot(eof[:, 1], lev)
plt.gca().invert_yaxis()
output_dict = {
    'lev': lev,
    'EOF': eof
}
# %% ================== Part 4: save data ==================== #
# path for saving data
path_save: str = f'/work/b11209013/2024_Research/MPAS/PC/'
jl.dump(output_dict, f'{path_save}EOF.joblib', compress=3)

# %%
