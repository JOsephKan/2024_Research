# This program is to composite the longwave and shortwave profiles from MPAS output

# import packages
import numpy as np
from netCDF4 import Dataset

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load Dataset
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/CNTL/"

## MPAS Dataset
dims: dict[str, np.ndarray] = dict()
data: dict[str, np.ndarray] = dict()

### LW data
with Dataset(f"{path}rthratenlw.nc", "r", mmap=True) as f:
    for key in f.dimensions.items():
        dims[key[0]] = f.variables[key[0]][:]
    
    lat_lim = np.where((dims['lat'] >= -5) & (dims['lat'] <= 5))[0]

    for key in f.variables.keys():
        if not key in dims.keys():
            data[key] = f.variables[key][:, :, lat_lim, :].mean(axis=2)*86400

### SW data
with Dataset(f"{path}rthratensw.nc", "r", mmap=True) as f:
    for key in f.variables.keys():
        if not key in dims.keys():
            data[key] = f.variables[key][:, :, lat_lim, :].mean(axis=2)*86400

### Load selected events
lon_ref, time_ref = np.load(
    f"/home/b11209013/2024_Research/MPAS/Composite/Q1_event_sel/Q1_sel/CNTL.npy"
    )

time_itv = [
    np.linspace(time_ref[i]-16, time_ref[i]+16, 33).astype(int)
    for i in range(time_ref.size)
]

time_ticks = np.linspace(-4, 4, 33)

# compute anomaly
data_ano = dict(
    lw = data["rthratenlw"] - data["rthratenlw"].mean(axis=(0, 2))[None, :, None],
    sw = data["rthratensw"] - data["rthratensw"].mean(axis=(0, 2))[None, :, None]
)

# Filter out the selected data
data_sel = {
    key: np.array([
        data_ano[key][time_itv[i], :, lon_ref[i]]
        for i in range(time_ref.size)
    ]).mean(axis=0).T
    for key in data_ano.keys()
}

# plot out the composite
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'font.family': 'serif',
})

lev_cond = np.argmin(np.abs(dims['lev'] - 200))

plt.figure(figsize=(12, 7))
plt.contourf(
    time_ticks, dims['lev'][:lev_cond+1],
    data_sel['lw'][:lev_cond+1],
    cmap="RdBu_r",
    levels=np.linspace(-4, 4, 17),
    extend="both",
    norm=TwoSlopeNorm(vcenter=0),
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
plt.yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
plt.xlim(4, -4)
plt.ylim(1000, 100)
plt.xlabel("Relative Time [days]")
plt.ylabel("Level [hPa]")
plt.title("MPAS CNTL Composite Longwave heating Profile")

plt.text(2.3, 1200, 'Day After')
plt.text(-1.7, 1200, 'Day Before')
plt.colorbar()
plt.savefig("/home/b11209013/2024_Research/MPAS/Composite/raw_composite/Raw_comp_image/MPAS_sourced/CNTL_lw_raw_comp.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 7))
plt.contourf(
    time_ticks, dims['lev'][:lev_cond+1],
    data_sel['sw'][:lev_cond+1],
    cmap="RdBu_r",
    levels=np.linspace(-4, 4, 17),
    extend="both",
    norm=TwoSlopeNorm(vcenter=0),
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
plt.yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
plt.xlim(4, -4)
plt.ylim(1000, 100)
plt.xlabel("Relative Time [days]")
plt.ylabel("Level [hPa]")
plt.title("MPAS CNTL Composite Shortwave heating Profile")
plt.text(2.3, 1200, 'Day After')
plt.text(-1.7, 1200, 'Day Before')
plt.colorbar()
plt.savefig("/home/b11209013/2024_Research/MPAS/Composite/raw_composite/Raw_comp_image/MPAS_sourced/CNTL_sw_raw_comp.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 7))
plt.contourf(
    time_ticks, dims['lev'][:lev_cond+1],
    data_sel['lw'][:lev_cond+1] + data_sel['sw'][:lev_cond+1],
    cmap="RdBu_r",
    levels=np.linspace(-4, 4, 17),
    extend="both",
    norm=TwoSlopeNorm(vcenter=0),
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.yscale('log')
plt.yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10).astype(int))
plt.xlim(4, -4)
plt.ylim(1000, 100)
plt.xlabel("Relative Time [days]")
plt.ylabel("Level [hPa]")
plt.title("MPAS CNTL Composite Radiative heating Profile")
plt.text(2.3, 1200, 'Day After')
plt.text(-1.7, 1200, 'Day Before')
plt.colorbar()
plt.savefig("/home/b11209013/2024_Research/MPAS/Composite/raw_composite/Raw_comp_image/MPAS_sourced/CNTL_qr_raw_comp.png", dpi=300)
plt.show()


