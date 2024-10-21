# This program is to composite first and second baroclinic mode profile
# import packages
import sys
import numpy as np
import pickle as pkl
import netCDF4 as nc
from joblib import Parallel, delayed, load
from matplotlib import pyplot as plt

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# ======================== #
# load data
pc_path  = '/home/b11209013/2024_Research/MPAS/GrowthRate/PCA_file/'
lrf_path = '/home/b11209013/2024_Research/MPAS/LRF/LRF_file/'

## q1 data
with nc.Dataset(f'/work/b11209013/MPAS/merged_data/CNTL/q1.nc', 'r', mmap=True) as f:
    lat = f.variables['lat'][:]
    lev = f.variables['lev'][:]

    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
    q1 = f.variables['q1'][:, :, lat_lim, :]

pc_dict = {}

## PCA file
with nc.Dataset(f'{pc_path}CNTL_PC.nc', mmap=True) as data:
    lon   = data["lon"][:]
    lat   = data["lat"][:]
    time  = data["time"][:]
    pc_dict['tpc1']  = data['tpc1'][:]
    pc_dict['tpc2']  = data['tpc2'][:]
    pc_dict['qvpc1'] = data['qvpc1'][:]
    pc_dict['qvpc2'] = data['qvpc2'][:]
    pc_dict['q1pc1'] = data['q1pc1'][:]
    pc_dict['q1pc2'] = data['q1pc2'][:]

## EOF
eof  = np.loadtxt('/home/b11209013/2024_Research/MPAS/GrowthRate/PCA_file/CNTL_EOF.txt')
eof1 = eof[:, 0]
eof2 = eof[:, 1]

## CNTL LRF
LRF = load(f"{lrf_path}lrf_CNTL.pkl")

lw_lrf = LRF["lw_lrf"]
sw_lrf = LRF["sw_lrf"]
cu_lrf = LRF["cu_lrf"]

# ========================= #
# select Kelvin wave events
## vertical integrate Q1
q1_ave = (q1[:, 1:] + q1[:, :-1])/2
q1_vint = -np.sum(q1_ave*np.diff(lev)[None, :, None, None]*100, axis=1)*86400/9.8/2.5e6

## format data
fmt = dp.Format(lat)

q1_sym = fmt.sym(q1_vint)

q1_fft = np.fft.fftshift(np.fft.fft2(q1_sym))[:, ::-1]

## select Kelvin wave band
wn = np.linspace(-360, 360, 720)
fr = np.linspace(-2, 2, 376)
wnm, frm = np.meshgrid(wn, fr)

cond = np.logical_or.reduce([
   wnm < 1,
   wnm > 14,
   frm < 1 / 20,
   frm > 1 / 2.5,
])

q1_fft[cond] *= 0

q1_filt = q1_fft*2

q1_sel_kel = np.fft.ifft2(np.fft.ifftshift(q1_filt))

ref_lon = np.argmin(np.abs(lon-240))

ref_tseries = q1_sel_kel[:, ref_lon]

pos_idx = np.where(ref_tseries > ref_tseries.std())[0]

time_sel = [np.arange(i-2, i+2.2, 1, dtype=int) for i in pos_idx]

# ========================== #
# select positive events
pc_sel = {}

for var in pc_dict.keys():

    pc_sel[var] = np.empty((len(pos_idx), 5, len(lat)))

    for i, t in enumerate(time_sel):
        for k, j in enumerate(t):
            pc_sel[var][i, k, :] = pc_dict[var][j, :, ref_lon]

    print(pc_sel[var].shape)
