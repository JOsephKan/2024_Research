# This program is to select the signigicant event for CCKW via bandpass filtered Q1 data
# %% import package
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

from matplotlib import pyplot as plt

from cartopy import crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

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
# date compute (compositing date: 2006~2017)
## setting total period of ERA5 data
date_tot = pd.date_range(start='1979-01-01', periods=15706, freq='d')

date_str: int = np.where((date_tot.year==2006))[0][0] # 2006/01/01
date_trm: int = np.where((date_tot.year==2017))[0][-1] # 2017/12/31

with nc.Dataset('/work/b11209013/2024_Research/nstcCCKW/Q1/Q1Flt.nc', 'r') as f:
    lon : np.ndarray = f.variables['lon'][:]
    lat : np.ndarray = f.variables['lat'][:]
    lev : np.ndarray = f.variables['lev'][:]
    time: np.ndarray = f.variables['time'][:]

    q1       = f.variables['Q1'][date_str:date_trm+1]

ltime, llev, llat, llon = q1.shape

# %% ================== Part 2: Vertical integrate Q1 ==================== #    
q1_ave = (q1[:, 1:] + q1[:, :-1]) / 2
q1_int = -np.sum(q1_ave*np.diff(lev)[None, :, None, None], axis=1) *86400/9.8/2.5e6
print(q1_int[0])
# %% ================== Part 3: Processing data ==================== #
## bandpass filter
wn = np.fft.fftfreq(llon, d = 1/llon).astype(int) # unit: None
fr = np.fft.fftfreq(ltime, d = 1) # unit: CPD

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
q1_fft = np.empty((llat, ltime, llon), dtype=complex)

for i in range(llat):
    ft = np.array([np.fft.fft(q1_int[j, i, :]) for j in range(ltime)])
    q1_fft[i] = np.array([np.fft.ifft(ft[:, j]) for j in range(llon)]).T

## Filt with the filter
mask = np.where(cond1 | cond2, 1, 0)

q1_recon = np.empty((llat, ltime, llon), dtype=complex)

for i in range(llat):
    masked = mask * q1_fft[i]
    rc = np.array([np.fft.fft(masked[:, j]) for j in range(llon)]).T
    q1_recon[i] = np.array([np.fft.ifft(rc[j]) for j in range(ltime)])

q1_recon = np.real(q1_recon).transpose(1, 0, 2)
print(q1_recon[0])
kel_var = np.var(q1_recon, axis=0)
print(kel_var.shape)

# %%

plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(
    1,
    1,
    figsize=(15, 4),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
)

ax.coastlines(linewidth=0.5)
ax.set_extent([-180, 180, -15, 15], crs=ccrs.PlateCarree())

c = ax.contourf(
    lon, lat, kel_var, transform=ccrs.PlateCarree(), cmap="terrain_r", extend="max",
    levels=np.linspace(1, 10, 10)
)
ax.set_xticks(np.linspace(-180, 180, 13), crs=ccrs.PlateCarree())
ax.set_yticks(np.linspace(-15, 15, 7), crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(r"Variance of $Q_1$ of Reconstructed Kelvin Wave")
ax.grid(color="black", linestyle=":", linewidth=0.5)
cbar = plt.colorbar(
    c, ax=ax, orientation="horizontal", label=r"$mm^2/day^2$", shrink=0.4, aspect=50
)
cbar.set_ticks(np.linspace(1, 10, 10))
plt.tight_layout()
plt.savefig('kel_var.png', dpi=300)
plt.show()
# %%
