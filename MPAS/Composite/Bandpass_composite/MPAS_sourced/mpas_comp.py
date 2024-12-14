# This program is to composite the MPAS LW, SW, Cu heating data with the temperature and moisture
# %% import package
import os
import re
import sys
import numpy as np
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
case = 'NSC'

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
dims: dict[str, np.ndarray] = dict()
data: dict[str, dict[str, np.ndarray]] = dict(pc1= dict(), pc2= dict())

with nc.Dataset(f"/work/b11209013/2024_Research/MPAS/PC/{case}_PC.nc", 'r', mmap=True) as f:
    for key in f.dimensions.items():
        dims[key[0]] = f.variables[key[0]][:]

    for key in f.variables.items():
        if not key[0] in dims.keys():
            data["pc1"][key[0]] = f.variables[key[0]][0].transpose(2, 1, 0)
            data["pc2"][key[0]] = f.variables[key[0]][1].transpose(2, 1, 0)

            # data shape: (time, lat, lon)



ltime, llat, llon = data['pc1']['t'].shape

## load EOF dataset 
with nc.Dataset("/work/b11209013/2024_Research/MPAS/PC/EOF.nc", 'r', mmap=True) as f:
    lev : np.ndarray = f.variables['lev'][:]
    eof1: np.ndarray = f.variables['EOF'][0][:, None]
    eof2: np.ndarray = f.variables['EOF'][1][:, None]

# load selected data
sel = np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/Q1_sel/{case}.npy')
lon_ref : np.ndarray = sel[0]
time_ref: np.ndarray = sel[1]


# %% ================== Part 2: Apply bandpass filter ==================== #

# symmetrize the data
fmt = dp.Format(dims["lat"])

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

sw_1st = np.matmul(eof1, data_sel['pc1']['sw'])
sw_2nd = np.matmul(eof2, data_sel['pc2']['sw'])
sw_tot = sw_1st + sw_2nd

cu_1st = np.matmul(eof1, data_sel['pc1']['cu'])
cu_2nd = np.matmul(eof2, data_sel['pc2']['cu'])
cu_tot = cu_1st + cu_2nd

heating_sum = lw_tot + sw_tot + cu_tot

t_mean = np.sum(
    np.multiply(((t_tot[1:]+t_tot[:-1])/2), (np.diff(lev)[..., None])*100),
    axis=0
) / np.sum(np.diff(lev)*100)

lw_mean = np.sum(
    np.multiply((lw_tot[1:]+lw_tot[:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)

sw_mean = np.sum(
    np.multiply((sw_tot[1:]+sw_tot[:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)

cu_mean = np.sum(
    np.multiply((cu_tot[1:]+cu_tot[:-1])/2, np.diff(lev)[:, None]*100),
    axis=0
) / np.sum(np.diff(lev)*100, axis=0)


# %%
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

# First subplot with contourf
ax1 = plt.subplot(gs[0])
cr1 = ax1.contourf(
    time_ticks, lev[:-2], heating_sum[:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax1.contour(
    time_ticks, lev[:-2], t_tot[:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = plt.contour(
    time_ticks, lev[:-2], q_tot[:-2],
    levels=[-0.4, -0.3, -0.2,  0.2, 0.3, 0.4],
    colors='forestgreen', linewidths=1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
ax1.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
ax1.set_xlim(4, -4)
ax1.set_ylim(1000, 100)
ax1.set_ylabel("Level [hPa]")
plt.clabel(c1, inline=True, fontsize=8)

# Second subplot with line plot
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(
    time_ticks, lw_mean,
    color='royalblue',
    label='LW'
)
ax2.plot(
    time_ticks, sw_mean,
    color='sienna',
    label='SW'
)
ax2.plot(
    time_ticks, cu_mean,
    color='forestgreen',
    label='Cu'
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax2.set_xlim(4, -4)
ax2.set_ylim(-4, 5)
ax2.set_xticks(np.linspace(4, -4, 9))
ax2.set_yticks([-3, -1.5, 0, 1.5, 3])
ax2.set_ylabel('K/day')

cax = inset_axes(ax1, width="1.5%", height="80%", loc="center right", bbox_to_anchor=(0.04, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)


ax3 = ax2.twinx()
ax3.plot(
    time_ticks, t_mean,  # Plot data along the first row as an example
    color='k', 
    label='T'
)
plt.gca().spines['top'].set_visible(False)
ax3.set_ylabel('K')
ax3.set_ylim(-0.34, 0.34)
ax3.set_yticks(np.linspace(-0.4, 0.4, 5))

plt.text(2.3, -0.8, 'Day After')
plt.text(-1.7, -0.8, 'Day Before')

cbar = fig.colorbar(
    cr1, cax=cax, ax=[ax1, ax2], location="right",  label="K/day")
cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6, 8])

ax1.set_title(f'Case: {case}, Heating: MPAS, Bandpass Filter: Yes\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen)',
fontsize=10, loc='left')
plt.savefig(f'/home/b11209013/2024_Research/MPAS/Composite/Bandpass_composite/Filtered_comp_image/MPAS_sourced/{case}_MPAS_comp.png', dpi=300)
plt.show()
# %%
