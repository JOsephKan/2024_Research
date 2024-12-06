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

exp1      : str = 'CNTL'
exp2      : str = 'NSC'

exp1_fname    : str = f'{path}{exp1}_heating.joblib'
exp2_fname: str = f'{path}{exp2}_heating.joblib'

exp1_data = jl.load(exp1_fname)
exp2_data = jl.load(exp2_fname)

# 2. reshape data
lev       : np.ndarray = exp1_data['lev']
time_ticks: np.ndarray = exp1_data['time_tick']

data: dict[str, dict[str, np.ndarray]] = {
    'exp1':{
        'pc1':{
            't' : exp1_data['vert_prof']['pc1']['t'],
            'qv': exp1_data['vert_prof']['pc1']['qv'],
            'q1': exp1_data['vert_prof']['pc1']['q1'],
            'lw': exp1_data['pc1']['lw'],
            'sw': exp1_data['pc1']['sw'],
            'cu': exp1_data['pc1']['cu'][:38],
        },
        'pc2':{
            't' : exp1_data['vert_prof']['pc2']['t'],
            'qv': exp1_data['vert_prof']['pc2']['qv'],
            'q1': exp1_data['vert_prof']['pc2']['q1'],
            'lw': exp1_data['pc2']['lw'],
            'sw': exp1_data['pc2']['sw'],
            'cu': exp1_data['pc2']['cu'][:38],
        }
    },
    'exp2':{
        'pc1':{
            't' : exp2_data['vert_prof']['pc1']['t'],
            'qv': exp2_data['vert_prof']['pc1']['qv'],
            'q1': exp2_data['vert_prof']['pc1']['q1'],
            'lw': exp2_data['pc1']['lw'],
            'sw': exp2_data['pc1']['sw'],
            'cu': exp2_data['pc1']['cu'][:38],
        },
        'pc2':{
            't' : exp2_data['vert_prof']['pc2']['t'],
            'qv': exp2_data['vert_prof']['pc2']['qv'],
            'q1': exp2_data['vert_prof']['pc2']['q1'],
            'lw': exp2_data['pc2']['lw'],
            'sw': exp2_data['pc2']['sw'],
            'cu': exp2_data['pc2']['cu'][:38],
        }
    },
}

del exp1_data, exp2_data

# %% Porcessing data
## 1. compute sum of leading modes
data_sum: dict[str, np.ndarray] = {
    'exp1':{
        't' : data['exp1']['pc1']['t']  + data['exp1']['pc2']['t'],
        'qv': data['exp1']['pc1']['qv'] + data['exp1']['pc2']['qv'],
        'q1': data['exp1']['pc1']['q1'] + data['exp1']['pc2']['q1'],
        'lw': data['exp1']['pc1']['lw'] + data['exp1']['pc2']['lw'],
        'sw': data['exp1']['pc1']['sw'] + data['exp1']['pc2']['sw'],
        'cu': data['exp1']['pc1']['cu'] + data['exp1']['pc2']['cu'],
    },
    'exp2':{
        't' : data['exp2']['pc1']['t']  + data['exp2']['pc2']['t'],
        'qv': data['exp2']['pc1']['qv'] + data['exp2']['pc2']['qv'],
        'q1': data['exp2']['pc1']['q1'] + data['exp2']['pc2']['q1'],
        'lw': data['exp2']['pc1']['lw'] + data['exp2']['pc2']['lw'],
        'sw': data['exp2']['pc1']['sw'] + data['exp2']['pc2']['sw'],
        'cu': data['exp2']['pc1']['cu'] + data['exp2']['pc2']['cu'],
    },
}

data_vint: dict[str, dict[str, np.ndarray]] = {
    'exp1':{
        't' : vert_int(data_sum['exp1']['t'], lev),
        'q1': vert_int(data_sum['exp1']['q1'], lev),
        'lw': vert_int(data_sum['exp1']['lw'], lev),
        'sw': vert_int(data_sum['exp1']['sw'], lev),
        'cu': vert_int(data_sum['exp1']['cu'], lev),
    },
    'exp2':{
        't' : vert_int(data_sum['exp1']['t'], lev),
        'q1': vert_int(data_sum['exp1']['q1'], lev),
        'lw': vert_int(data_sum['exp1']['lw'], lev),
        'sw': vert_int(data_sum['exp1']['sw'], lev),
        'cu': vert_int(data_sum['exp1']['cu'], lev),
    },
}

total_heating: dict[str, np.ndarray] = {
    'exp1': data_sum['exp1']['lw'] + data_sum['exp1']['sw'] + data_sum['exp1']['cu'],
    'exp2': data_sum['exp2']['lw'] + data_sum['exp2']['sw'] + data_sum['exp2']['cu'],
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
    time_ticks, lev[:-2], total_heating['exp1'][:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax1.contour(
    time_ticks, lev[:-2], data_sum['exp1']['t'][:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = ax1.contour(
    time_ticks, lev[:-2], data_sum['exp1']['qv'][:-2],
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
    time_ticks, data_vint['exp1']['lw'],
    color='royalblue',
    label='LW'
)
ax2.plot(
    time_ticks, data_vint['exp1']['sw'],
    color='sienna',
    label='SW'
)
ax2.plot(
    time_ticks, data_vint['exp1']['cu'],
    color='forestgreen',
    label='Cu'
)
ax2.plot(
    time_ticks, data_vint['exp1']['q1'],
    color='forestgreen',
    linestyle='--',
    label='Q1'
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
    time_ticks, data_vint['exp1']['t'],  # Plot data along the first row as an example
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

ax1.set_title(f'Case: CNTL (LRF)\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen), Q1 (forestgreen dashed)',
fontsize=10, loc='left')


ax4 = plt.subplot(gs[0, 1])
cr1 = ax4.contourf(
    time_ticks, lev[:-2], total_heating['exp2'][:-2],
    cmap='RdBu_r',
    levels=np.linspace(-6, 8, 29),
    extend='both',
    norm=TwoSlopeNorm(vcenter=0),
)
c1 = ax4.contour(
    time_ticks, lev[:-2], data_sum['exp2']['t'][:-2],
    colors='k',
    linewidths=1,
    levels=[-1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25]
)
crqv = ax4.contour(
    time_ticks, lev[:-2], data_sum['exp2']['qv'][:-2],
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
    time_ticks, data_vint['exp2']['lw'],  
    color='royalblue',
    label='LW'
)
ax5.plot(
    time_ticks, data_vint['exp2']['sw'],
    color='sienna',
    label='SW'
)
ax5.plot(
    time_ticks, data_vint['exp2']['cu'],
    color='forestgreen',
    label='Cu'
)
ax5.plot(
    time_ticks, data_vint['exp2']['q1'],
    color='forestgreen',
    linestyle='--',
    label='Q1'
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
    time_ticks, data_vint['exp2']['t'],  # Plot data along the first row as an example
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

ax4.set_title(f'Case: {exp2} (LRF)\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen), Q1 (forestgreen dashed)',
fontsize=10, loc='left')


plt.savefig(f'{image_path}CNTL_vs_{exp2}.png', dpi=300)

# %%
