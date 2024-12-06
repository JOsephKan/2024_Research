# This program is to generate the comparison of the filtered image and the original image
# %% import package
import os 
import sys
import glob
import numpy as np
import joblib as jl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# %% design functions
## loading data
def load_data(
        fname: str,
) -> dict[str, dict[str, np.ndarray]]:
    data: np.ndarray = jl.load(fname)

    return data

## vertical averaged
def vert_int(
        data: np.ndarray,
        lev : np.ndarray,
)-> np.ndarray:
    data_vint: np.ndarray = np.sum(
        np.multiply(((data[1:]+data[:-1])/2), (np.diff(lev)[..., None])*100),
        axis=0
    ) / np.sum(np.diff(lev)*100)

    return data_vint
# %% load data
# 1. load the original data for heating
path      : str = '/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/LRF_com_heating/'
image_path: str = '/home/b11209013/2024_Research/MPAS/Composite/Image/'

exp       : str = 'NSC'

filted_fname    : str = f'{path}{exp}_heating.joblib'
non_filted_fname: str = f'{path}{exp}_heating_no_filt.joblib'

non_filted_data = jl.load(non_filted_fname)
filted_data     = jl.load(filted_fname)

# 2. reshape data
lev: np.ndarray = non_filted_data['lev']
time_ticks: np.ndarray = non_filted_data['time_tick']

data: dict[str, dict[str, np.ndarray]] = {
    'non_filted':{
        'pc1':{
            't' : non_filted_data['vert_prof']['pc1']['t'],
            'qv': non_filted_data['vert_prof']['pc1']['qv'],
            'q1': non_filted_data['vert_prof']['pc1']['q1'],
            'lw': non_filted_data['pc1']['lw'],
            'sw': non_filted_data['pc1']['sw'],
            'cu': non_filted_data['pc1']['cu'][:38],
        },
        'pc2':{
            't' : non_filted_data['vert_prof']['pc2']['t'],
            'qv': non_filted_data['vert_prof']['pc2']['qv'],
            'q1': non_filted_data['vert_prof']['pc2']['q1'],
            'lw': non_filted_data['pc2']['lw'],
            'sw': non_filted_data['pc2']['sw'],
            'cu': non_filted_data['pc2']['cu'][:38],
        }
    },
    'filted':{
        'pc1':{
            't' : filted_data['vert_prof']['pc1']['t'],
            'qv': filted_data['vert_prof']['pc1']['qv'],
            'q1': filted_data['vert_prof']['pc1']['q1'],
            'lw': filted_data['pc1']['lw'],
            'sw': filted_data['pc1']['sw'],
            'cu': filted_data['pc1']['cu'][:38],
        },
        'pc2':{
            't' : filted_data['vert_prof']['pc2']['t'],
            'qv': filted_data['vert_prof']['pc2']['qv'],
            'q1': filted_data['vert_prof']['pc2']['q1'],
            'lw': filted_data['pc2']['lw'],
            'sw': filted_data['pc2']['sw'],
            'cu': filted_data['pc2']['cu'][:38],
        }
    },
}

del non_filted_data, filted_data

# %% Porcessing data
## 1. compute sum of leading modes
data_sum: dict[str, np.ndarray] = {
    'non_filted':{
        't' : data['non_filted']['pc1']['t']  + data['non_filted']['pc2']['t'],
        'qv': data['non_filted']['pc1']['qv'] + data['non_filted']['pc2']['qv'],
        'q1': data['non_filted']['pc1']['q1'] + data['non_filted']['pc2']['q1'],
        'lw': data['non_filted']['pc1']['lw'] + data['non_filted']['pc2']['lw'],
        'sw': data['non_filted']['pc1']['sw'] + data['non_filted']['pc2']['sw'],
        'cu': data['non_filted']['pc1']['cu'] + data['non_filted']['pc2']['cu'],
    },
    'filted':{
        't' : data['filted']['pc1']['t']      + data['filted']['pc2']['t'],
        'qv': data['filted']['pc1']['qv']     + data['filted']['pc2']['qv'],
        'q1': data['filted']['pc1']['q1']     + data['filted']['pc2']['q1'],
        'lw': data['filted']['pc1']['lw']     + data['filted']['pc2']['lw'],
        'sw': data['filted']['pc1']['sw']     + data['filted']['pc2']['sw'],
        'cu': data['filted']['pc1']['cu']     + data['filted']['pc2']['cu'],
    },
}

data_vint: dict[str, dict[str, np.ndarray]] = {
    'non_filted':{
        't' : vert_int(data_sum['non_filted']['t'], lev),
        'q1': vert_int(data_sum['non_filted']['q1'], lev),
        'lw': vert_int(data_sum['non_filted']['lw'], lev),
        'sw': vert_int(data_sum['non_filted']['sw'], lev),
        'cu': vert_int(data_sum['non_filted']['cu'], lev),
    },
    'filted':{
        't' : vert_int(data_sum['filted']['t'], lev),
        'q1': vert_int(data_sum['filted']['q1'], lev),
        'lw': vert_int(data_sum['filted']['lw'], lev),
        'sw': vert_int(data_sum['filted']['sw'], lev),
        'cu': vert_int(data_sum['filted']['cu'], lev),
    },
}

total_heating: dict[str, np.ndarray] = {
    'non_filted': data_sum['non_filted']['lw'] + data_sum['non_filted']['sw'] + data_sum['non_filted']['cu'],
    'filted': data_sum['filted']['lw'] + data_sum['filted']['sw'] + data_sum['filted']['cu'],
}

# %% plot data
## 1. plot setting
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

## 2. plot data
fig = plt.figure(figsize=(24, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])

# 1. plot out the top panel
ax1 = plt.subplot(gs[0, 0])
cr1 = ax1.contourf(
    time_ticks, lev[:-2], total_heating['filted'][:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax1.contour(
    time_ticks, lev[:-2], data_sum['filted']['t'][:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = ax1.contour(
    time_ticks, lev[:-2], data_sum['filted']['qv'][:-2],
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
ax2 = plt.subplot(gs[1, 0], sharex=ax1)
ax2.plot(
    time_ticks, data_vint['filted']['q1'],
    color='forestgreen',
    linestyle='--',
    label='Q1'
)
ax2.plot(
    time_ticks, data_vint['filted']['lw'],  
    color='royalblue',
    label='LW'
)
ax2.plot(
    time_ticks, data_vint['filted']['sw'],
    color='sienna',
    label='SW'
)
ax2.plot(
    time_ticks, data_vint['filted']['cu'],
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
    time_ticks, data_vint['filted']['t'],  # Plot data along the first row as an example
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

ax1.set_title(f'Case: CNTL (LRF, filted)\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen), Q1 (forestgreen, dashed)',
fontsize=10, loc='left')


ax4 = plt.subplot(gs[0, 1])
cr1 = ax4.contourf(
    time_ticks, lev[:-2], total_heating['non_filted'][:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax4.contour(
    time_ticks, lev[:-2], data_sum['non_filted']['t'][:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = ax4.contour(
    time_ticks, lev[:-2], data_sum['non_filted']['qv'][:-2],
    levels=[-0.4, -0.3, -0.2,  0.2, 0.3, 0.4],
    colors='forestgreen', linewidths=1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
ax4.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
ax4.set_xlim(4, -4)
ax4.set_ylim(1000, 100)
ax4.set_ylabel("Level [hPa]")
plt.clabel(c1, inline=True, fontsize=8)

# Second subplot with line plot
ax5 = plt.subplot(gs[1, 1], sharex=ax4)
ax5.plot(
    time_ticks, data_vint['non_filted']['q1'],
    color='forestgreen',
    linestyle='--',
    label='Q1'
)
ax5.plot(
    time_ticks, data_vint['non_filted']['lw'],  
    color='royalblue',
    label='LW'
)
ax5.plot(
    time_ticks, data_vint['non_filted']['sw'],
    color='sienna',
    label='SW'
)
ax5.plot(
    time_ticks, data_vint['non_filted']['cu'],
    color='forestgreen',
    label='Cu'
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax5.set_xlim(4, -4)
ax5.set_ylim(-4, 5)
ax5.set_xticks(np.linspace(4, -4, 9))
ax5.set_yticks([-3, -1.5, 0, 1.5, 3])
ax5.set_ylabel('K/day')

cax = inset_axes(ax4, width="1.5%", height="80%", loc="right", bbox_to_anchor=(0.04, 0, 1, 1), bbox_transform=ax4.transAxes, borderpad=0)


ax6 = ax5.twinx()
ax6.plot(
    time_ticks, data_vint['non_filted']['t'],  # Plot data along the first row as an example
    color='k', 
    label='T'
)
plt.gca().spines['top'].set_visible(False)
ax6.set_ylabel('K')
ax6.set_ylim(-0.34, 0.34)
ax6.set_yticks(np.linspace(-0.4, 0.4, 5))

plt.text(2.3, -0.8, 'Day After')
plt.text(-1.7, -0.8, 'Day Before')

cbar = fig.colorbar(cr1, cax=cax, ax=[ax4, ax5], location="right",  label="K/day")
cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6, 8])

ax4.set_title(f'Case: CNTL (LRF, non_filted)\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen), Q1 (forestgreen, dashed)',
fontsize=10, loc='left')


plt.savefig(f'{image_path}{exp}_filt_vs_non_filt.png', dpi=300)

# %%
