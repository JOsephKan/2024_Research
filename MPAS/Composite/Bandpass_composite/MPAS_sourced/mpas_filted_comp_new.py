# this program is to construct # This program is to compute the compositing of CNTL
# %% section 1
# import package
import os
import sys
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append('/home/b11209013/Package/')
import Theory as th # type: ignore
import DataProcess as dp # type: ignore
import SignalProcess as sp # type: ignore

# %%
# Functions
# 1. FFT2
def fft2(
        data: np.ndarray,
) -> np.ndarray:
    fft = np.array([np.fft.fft(data[i]) for i in range(data.shape[0])])
    fft = np.array([np.fft.ifft(fft[:, i]) for i in range(fft.shape[1])]).T
    return fft

# 2. IFFT2
def ifft2(
        data: np.ndarray,
) -> np.ndarray:
    ifft = np.array([np.fft.fft(data[:, i]) for i in range(data.shape[1])]).T
    ifft = np.array([np.fft.ifft(ifft[i]) for i in range(ifft.shape[0])])
    return ifft.real

# 3. Mask
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

# vint
## vertical averaged
def vert_int(
        data: np.ndarray,
        lev : np.ndarray
) -> np.ndarray:
    
    data_ave: np.ndarray = (data[1:] + data[:-1])/2
    data_int: np.ndarray = np.sum(
        np.multiply(data_ave, np.diff(lev)[:, None]*100),
        axis = 0
    ) / np.sum(np.diff(lev)*100)
    
    return data_int

# %% section 2
# load data
# case name
#case = sys.argv[1]
case='CNTL'
# path
fname = f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.nc'

# load principal component data
with nc.Dataset(fname, 'r') as f:
    lon  = f['lon'][:]
    lat  = f['lat'][:]
    time = f['time'][:]

    data = {"pc1": {}, "pc2": {}}
    for var in ["t", "qv", "q1"]:
        data["pc1"][var] = f.variables[var][0].transpose(2, 1, 0)
        data["pc2"][var] = f.variables[var][1].transpose(2, 1, 0)

ltime, llat, llon = data['pc1']['t'].shape

# size of the pc: (time, lat, lon)

# load EOF structure

with nc.Dataset('/work/b11209013/2024_Research/MPAS/PC/EOF.nc', 'r') as f:
    lev = f['lev'][:]
    eof1 = f['EOF'][0]
    eof2 = f['EOF'][1]


# load LRF filei
with nc.Dataset(f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/{case}.nc', 'r') as f:
    lw  = np.where(np.isnan(f['lw'][:]) == True, 0, f['lw'][:])
    sw  = np.where(np.isnan(f['sw'][:]) == True, 0, f['sw'][:])
    cu  = np.where(np.isnan(f['cu'][:]) == True, 0, f['cu'][:])
    tot = np.where(np.isnan(f['tot'][:]) == True, 0, f['tot'][:])

    
# load reference longitude and time for compositing
lon_ref, time_ref = list(
    np.load(f'/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/Q1_sel/{case}.npy')
    )
# %%
# format data
fmt = dp.Format(lat)

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fmt.sym(data[pc][var])
        for var in data[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %%
# FFT on the symmetric data
fft: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fft2(sym[pc][var])
        for var in sym[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %%
# Bandpass filter
wn = np.fft.fftfreq(llon, d=1/llon).astype(int)
fr = np.fft.fftfreq(ltime, d=1/4)

wnm, frm = np.meshgrid(wn, fr)
mask_arr = mask(wnm, frm)

reconstruct: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: ifft2(fft[pc][var] * mask_arr)
        for var in fft[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

# %%
# select data
time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(len(time_ref))
]

time_ticks = np.linspace(-4, 4, 33)

# data selection
data_sel: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: np.array([
            reconstruct[pc][var][time_itv[i], lon_ref[i]]
            for i in range(len(time_ref))
        ]).mean(axis=0)
        for var in reconstruct[pc].keys()
    }
    for pc in ['pc1', 'pc2']
}

#plt.plot(time_ticks, data_sel['pc1']['t'])
#plt.plot(time_ticks, data_sel['pc2']['t'])

# %% 
# Generate heating profile from LRF
# 1. construct vertical profile of t and qv
vert_prof: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: np.matrix(eof1).T @ np.matrix(data_sel['pc1'][var])
        for var in ['t', 'qv', 'q1']
    },
    'pc2': {
        var: np.matrix(eof2).T @ np.matrix(data_sel['pc2'][var])
        for var in ['t', 'qv', 'q1']
    }
}

#plt.contourf(time_ticks, lev, vert_prof['pc1']['t'] + vert_prof['pc2']['t'])
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
#plt.colorbar()

# 2. construct state vector
state_vec: dict[str, np.ndarray] = {
    'pc1': np.concatenate([vert_prof['pc1']['t'], vert_prof['pc1']['qv']], axis=0),
    'pc2': np.concatenate([vert_prof['pc2']['t'], vert_prof['pc2']['qv']], axis=0)
}

# 3. compute heating profile
heating: dict[str, dict[str, np.ndarray]] = {
    pc: {
        'lw': lw @ state_vec[pc],
        'sw': sw @ state_vec[pc],
        'cu': cu @ state_vec[pc],
        'tot': tot @ state_vec[pc]
    }
    for pc in ['pc1', 'pc2']
}

heating['vert_prof'] = vert_prof
heating['lev'] = lev
heating['time_tick'] = time_ticks

# %%
# save file
#jl.dump(heating, f'/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/LRF_com_heating/{case}_heating.joblib')

# %% plot
## sum of first two PCs
t_tot = vert_prof['pc1']['t'] + vert_prof['pc2']['t']
q_tot = vert_prof['pc1']['qv'] + vert_prof['pc2']['qv']
q1_tot = vert_prof['pc1']['q1'] + vert_prof['pc2']['q1']
lw_tot = heating['pc1']['lw'] + heating['pc2']['lw']
sw_tot = heating['pc1']['sw'] + heating['pc2']['sw']
cu_tot = heating['pc1']['cu'] + heating['pc2']['cu']
heating_sum = heating['pc1']['tot'] + heating['pc2']['tot']

## vertical integral
lw_mean = vert_int(lw_tot, lev)
sw_mean = vert_int(sw_tot, lev)
cu_mean = vert_int(cu_tot, lev)
t_mean = vert_int(t_tot, lev)
tot_mean = vert_int(heating_sum, lev)

## plot setting
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
ax2.plot(
    time_ticks, tot_mean,
    color='k',
    linestyle='--',
    label='Total'
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

ax1.set_title(f'Exp: {case}, Heating: MPAS, Bandpass Filter: Yes\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen)',
fontsize=10, loc='left')
plt.savefig(f'/home/b11209013/2024_Research/MPAS/Composite/Bandpass_composite/Filtered_comp_image/{case}_MPAS_comp_new.png', dpi=300)
plt.show()
