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
with nc.Dataset('/work/b11209013/2024_Research/MPAS/merged_data/NSC/q1.nc', 'r') as f:
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
plt.scatter(lon[710], time[16], color='g', s=10)
plt.scatter(lon[540], time[20], color='g', s=10)
plt.scatter(lon[650], time[99], color='g', s=10)
plt.scatter(lon[440], time[83], color='g', s=10)
plt.scatter(lon[650], time[130], color='g', s=10)
plt.scatter(lon[480], time[113], color='g', s=10)
plt.scatter(lon[160], time[96], color='g', s=10)
plt.scatter(lon[160], time[181], color='g', s=10)
plt.scatter(lon[460], time[226], color='g', s=10)
plt.scatter(lon[600], time[244], color='g', s=10)
plt.scatter(lon[420], time[318], color='g', s=10)


plt.title('Reconstructed Q1')
plt.colorbar(c)
plt.show()


# %% ================== Part 5: Save the data ==================== #
lon_ref = np.array([710, 540, 650, 440, 650, 480, 160, 160, 460, 600, 420]).astype(int)
time_ref = np.array([16, 20, 99, 83, 130, 113, 96, 181, 226, 244, 318]).astype(int)

output_list = [lon_ref, time_ref]

np.save('NSC.npy', output_list)