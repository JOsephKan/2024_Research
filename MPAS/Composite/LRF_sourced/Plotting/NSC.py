# import package
# %% section 1: import package
# import official package
import numpy as np
import joblib as jl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% section 2: load data
# Load LRF-generated heating profile
## file name
fname: str = '/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/LRF_com_heating/NSC_heating_no_filt.joblib'

## load data
data = jl.load(fname)

lev       : np.ndarray = data['lev']
time_ticks: np.ndarray = data['time_tick']
data: dict[str, dict[str, np.ndarray]] = {
    'pc1':{
        't' : data['vert_prof']['pc1']['t'],
        'qv': data['vert_prof']['pc1']['qv'],
        'lw': data['pc1']['lw'],
        'sw': data['pc1']['sw'],
        'cu': data['pc1']['cu'][:38],
    },
    'pc2':{
        't' : data['vert_prof']['pc2']['t'],
        'qv': data['vert_prof']['pc2']['qv'],
        'lw': data['pc2']['lw'],
        'sw': data['pc2']['sw'],
        'cu': data['pc2']['cu'][:38],
    }
}

# %% section 3: compute heating profile
# 1. Compute heating profile
t_tot : np.ndarray = data['pc1']['t']  + data['pc2']['t']
q_tot : np.ndarray = data['pc1']['qv'] + data['pc2']['qv']
lw_tot: np.ndarray = data['pc1']['lw'] + data['pc2']['lw']
sw_tot: np.ndarray = data['pc1']['sw'] + data['pc2']['sw']
cu_tot: np.ndarray = data['pc1']['cu'] + data['pc2']['cu']

heating_sum: np.ndarray = lw_tot + sw_tot + cu_tot

# 2. vertical average
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

# %% section 4: plot data
# plot heating profile
## 1. plot setting
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

## 2. plot data
fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

# 1. plot out the top panel
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

ax1.set_title(f'Case: NSC (LRF)\n\
Upper: Total Heating (Shading); Composite Temperature (Black Contour), Moisture (Green Contour)\n\
Lower: Column-integrated Temperature (k), LW (royalblue), SW (sienna), CU (forestgreen)',
fontsize=10, loc='left')
plt.savefig(f'/home/b11209013/2024_Research/MPAS/Composite/LRF_sourced/heating_image/NSC_heating_no_filt.png', dpi=300)

# %%
