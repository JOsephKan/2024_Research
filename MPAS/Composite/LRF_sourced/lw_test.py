# This program is to composite the MPAS LW, SW, Cu heating data with the temperature and moisture
# %% import package
import os
import re
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
## import sself-defined package
sys.path.append('/home/b11209013/Package')
import Theory as th         #type: ignore
import DataProcess as dp    #type: ignore
import SignalProcess as sp  #type: ignore

## accept the system parameter
case = 'NCRF'

# %% ================== Part 0: Define functions ==================== #
# Functions
## band pass filter mask
def mask(wnm, frm):
    """
    Checks if (wnm, frm) lies in either the positive or negative region.

    Parameters:
        wnm: The wavenumber.
        frm: The frequency.

    Returns:
        A tuple of booleans (is_in_positive_region, is_in_negative_region):
        - is_in_positive_region: True if (wnm, frm) lies in the positive region.
        - is_in_negative_region: True if (wnm, frm) lies in the negative region.
    """
    kel_curves = lambda ed, k: (86400/(2*np.pi*6.371e6))*np.sqrt(9.81*ed)*k

    is_in_positive_region = (
        (wnm >= 1) & (wnm <= 14) &
        (frm >= 1 / 20) & (frm <= 1 / 2.5) &
        (frm <= kel_curves(90, wnm)) &
        (frm >= kel_curves(8, wnm))
    )
    
    is_in_negative_region = (
        (wnm <= -1) & (wnm >= -14) &
        (frm <= -1 / 20) & (frm >= -1 / 2.5) &
        (frm >= kel_curves(90, wnm)) &
        (frm <= kel_curves(8, wnm))
    )
    
    return np.where(is_in_positive_region | is_in_negative_region, 1, 0)

## FFT2
def fft2(
    data: np.ndarray
) -> np.ndarray:
    fft = np.array([np.fft.fft(data[i]) for i in range(data.shape[0])])
    fft = np.array([np.fft.ifft(fft[:, i]) for i in range(data.shape[1])]).T

    return fft

## IFFT2
def ifft2(
    data: np.ndarray
) -> np.ndarray:
    ifft = np.array([np.fft.fft(data[:, i]) for i in range(data.shape[1])]).T
    ifft = np.array([np.fft.ifft(ifft[i]) for i in range(data.shape[0])])

    return ifft.real


# %% ================== Part 1: load data ==================== #
# path for case data
path: str = f'/work/b11209013/2024_Research/MPAS/merged_data/{case}/'

# Load PC dataset
PC_data = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.joblib')

lon : np.ndarray = PC_data['lon']
lat : np.ndarray = PC_data['lat']
time: np.ndarray = PC_data['time']
data: dict[str, np.ndarray] = PC_data['pc']

data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: data[var][0]
        for var in data.keys()
    },
    'pc2': {
        var: data[var][1]
        for var in data.keys()
    }
}

ltime, llat, llon = data['pc1']['lw'].shape

## load EOF dataset 
EOF_data = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/EOF.joblib')

lev     = EOF_data['lev']
eof     = EOF_data['EOF']

eof1 = np.matrix(eof[:, 0]).T
eof2 = np.matrix(eof[:, 1]).T

# load selected data
sel = np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/{case}.npy')
lon_ref : np.ndarray = sel[0]
time_ref: np.ndarray = sel[1]
# %% ================== Part 2: Apply bandpass filter ==================== #

# symmetrize the data
fmt = dp.Format(lat)

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fmt.sym(data[pc][var])
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# FFT on symmetrized data
fft: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fft2(sym[pc][var])
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# bandpass filter
wn = np.fft.fftfreq(llon, d=1/llon).astype(int)
fr = np.fft.fftfreq(ltime, d=1/4)

wnm, frm = np.meshgrid(wn, fr)
mask_arr = mask(wnm, frm)

filtered: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: mask_arr * fft[pc][var]
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# IFFT on filtered data
reconstruct: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: ifft2(filtered[pc][var])
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %% ================== Part 3: Select data ==================== #
# setting the compositing interval
time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(time_ref.size)
]

time_ticks = np.linspace(-4, 4, 33)

# select data
data_sel: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: np.matrix([
            reconstruct[pc][var][time_itv[i], lon_ref[i]]
            for i in range(len(time_ref))
        ]).mean(axis=0)
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

print(lev)

# %% ================== Part 4: Composite ==================== #
# composite the vertical profile with time ticks
t_1st  = np.matmul(eof1, data_sel['pc1']['t'])
t_2nd  = np.matmul(eof2, data_sel['pc2']['t'])
t_tot = t_1st + t_2nd

q_1st  = np.matmul(eof1, data_sel['pc1']['qv'])
q_2nd  = np.matmul(eof2, data_sel['pc2']['qv'])
q_tot = q_1st + q_2nd

q1_1st = np.matmul(eof1, data_sel['pc1']['q1'])
q1_2nd = np.matmul(eof2, data_sel['pc2']['q1'])
q1_tot = q1_1st + q1_2nd

lw_1st = np.matmul(eof1, data_sel['pc1']['lw'])
lw_2nd = np.matmul(eof2, data_sel['pc2']['lw'])
lw_tot = lw_1st + lw_2nd


# %%
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})
crlw = plt.contourf(time_ticks, lev, lw_tot, cmap='coolwarm', norm=TwoSlopeNorm(vcenter=0))
ct = plt.contour(time_ticks, lev, t_tot, colors='black', levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25])
cqv = plt.contour(time_ticks, lev, q_tot, colors='green', levels=[-0.4, -0.3, -0.2,  0.2, 0.3, 0.4])
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.clabel(ct, inline=True, fontsize=10)
plt.clabel(cqv, inline=True, fontsize=10)
plt.colorbar(crlw)

# %%
