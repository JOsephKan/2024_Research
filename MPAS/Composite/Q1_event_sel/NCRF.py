# This program is to select the signigicant event for CCKW via bandpass filtered Q1 data
# %% import package
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

from matplotlib import pyplot as plt

sys.path.append('/home/b11209013/Package')
import DataProcess as dp   #type: ignore
import SignalProcess as sp #type: ignore

# %% Functions
def is_within_region(wnm, frm, wnm_min, wnm_max, frm_min, frm_max, kel_sign=1):
    kel_curves = lambda ed, k: (86400/(2*np.pi*6.371e6))*np.sqrt(9.81*ed)*k
    
    return (
        (wnm > wnm_min) & (wnm < wnm_max) &
        (frm > kel_sign * frm_min) & (frm < kel_sign * frm_max) &
        (frm < kel_sign * kel_curves(90, wnm)) &
        (frm > kel_sign * kel_curves(8, wnm))
    )

# %% ================== Part 1: load data ==================== #
with nc.Dataset('/work/b11209013/2024_Research/MPAS/merged_data/NCRF/q1.nc', 'r') as f:
    lon : np.ndarray = f.variables['lon'][:]
    lat : np.ndarray = f.variables['lat'][:]
    lev : np.ndarray = f.variables['lev'][:]
    time: np.ndarray = f.variables['time'][:]

    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
    
    lat : np.ndarray = lat[lat_lim]
    q1       = f.variables['q1'][:, :, lat_lim, :]

ltime, llev, llat, llon = q1.shape

# %% ================== Part 2: Vertical integrate Q1 ==================== #    
q1_ave = (q1[:, 1:] + q1[:, :-1]) / 2
q1_int = -np.sum(q1_ave*np.diff(lev)[None, :, None, None]*100, axis=1) *86400/9.8/2.5e6

# %% ================== Part 3: Processing data ==================== #
## symmetrize the data
fmt = dp.Format(lat)

q1_sym = fmt.sym(q1_int)

## bandpass filter
wn = np.fft.fftfreq(llon, d = 1/llon).astype(int) # unit: None
fr = np.fft.fftfreq(ltime, d = 1/4) # unit: CPD

wnm, frm = np.meshgrid(wn, fr)


cond1 = is_within_region(
    wnm, frm,
    wnm_min=1, wnm_max=14,
    frm_min=1/20, frm_max=1/2.5
)

cond2 = is_within_region(
    wnm, frm,
    wnm_min=-14, wnm_max=-1,
    frm_min=1/20, frm_max=1/2.5,
    kel_sign=-1
)

## FFT on q1_sym
q1_fft = np.array([np.fft.fft(q1_sym[i]) for i in range(ltime)])
q1_fft = np.array([np.fft.ifft(q1_fft[:, i]) for i in range(llon)]).T

## Filt with the filter
mask = np.where(cond1 | cond2, 1, 0)
q1_filted = mask * q1_fft

## reconstruct
q1_recon = np.array([np.fft.fft(q1_filted[:, i]) for i in range(llon)]).T
q1_recon = np.array([np.fft.ifft(q1_recon[i]) for i in range(ltime)])

q1_recon = np.real(q1_recon)

# %% ================== Part 4: selecting positive events ==================== #
g99 = q1_recon.mean() + 2.29*q1_recon.std() # 99% z-test, single tailed

plt.figure(figsize=(8, 6))
c = plt.contourf(lon, time, q1_recon, levels=30, cmap='RdBu_r')
plt.contour(lon, time, q1_recon, levels=[g99], colors='k')
plt.scatter(lon[489], time[16], color='g', s=10)
plt.scatter(lon[200], time[36], color='g', s=10)
plt.scatter(lon[350], time[36], color='g', s=10)
plt.scatter(lon[200], time[64], color='g', s=10)
plt.scatter(lon[500], time[110], color='g', s=10)
plt.scatter(lon[380], time[142], color='g', s=10)
plt.scatter(lon[234], time[135], color='g', s=10)
plt.scatter(lon[550], time[235], color='g', s=10)
plt.scatter(lon[465], time[242], color='g', s=10)
plt.scatter(lon[380], time[297], color='g', s=10)
plt.scatter(lon[260], time[299], color='g', s=10)
plt.scatter(lon[80], time[272], color='g', s=10)
plt.scatter(lon[260], time[340], color='g', s=10)


plt.title('Reconstructed Q1')
plt.colorbar(c)
plt.show()


# %% ================== Part 5: Save the data ==================== #
lon_ref = np.array([489, 200, 350, 200, 500, 380, 234, 550, 465, 380, 260, 80, 260]).astype(int)
time_ref = np.array([16, 36, 36, 64, 110, 142, 135, 235, 242, 297, 299, 272, 340]).astype(int)

output_list = [lon_ref, time_ref]

np.save('NCRF.npy', output_list)