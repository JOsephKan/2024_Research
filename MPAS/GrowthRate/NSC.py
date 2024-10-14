import sys
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# ============ #
# Load data
case = "NSC"
path = f"/work/b11209013/MPAS/PC/{case}/"

data = {}

with nc.Dataset(f"{path}q1pc_from_q1eof.nc", "r") as f:
    lon, lat, time = f.variables["lon"][:], f.variables["lat"][:], f.variables["time"][:]
    data["q1pc1"] = f.variables["q1pc"][0] * 86400 / 1004.5
    data["q1pc2"] = f.variables["q1pc"][1] * 86400 / 1004.5

with nc.Dataset(f"{path}tpc_from_q1eof.nc", "r") as f:
    data["tpc1"] = f.variables["tpc"][0]
    data["tpc2"] = f.variables["tpc"][1]

# =============== #
# Format data
fmt = dp.Format(lat)

# Using dict comprehension to apply the sym() method
sym_data = {key: fmt.sym(data[key]) for key in data.keys()}

# Split data into chunks using NumPy array slicing
sym_split = {key: np.array([sym_data[key][i * 30: i * 30 + 96] for i in range(10)]) for key in sym_data.keys()}

# ================== #
# Cross-spectrum computation using parallel processing
ft = sp.Fourier()

def compute_cross_spectrum(q1pc, tpc):
    return {
        "QT": np.mean([ft.CrossSpectrum(q1pc[i], tpc[i]) for i in range(10)], axis=0),
        "TT": np.mean([ft.CrossSpectrum(tpc[i], tpc[i]) for i in range(10)], axis=0)
    }

# Use Parallel processing for both pc1 and pc2
sym_prop = Parallel(n_jobs=-1)(delayed(compute_cross_spectrum)(sym_split[f"q1pc{i}"], sym_split[f"tpc{i}"]) for i in [1, 2])

# Extract results
growth_rate = {
    "pc1": 2 * np.real(sym_prop[0]["QT"]) / np.real(sym_prop[0]["TT"]),
    "pc2": 2 * np.real(sym_prop[1]["QT"]) / np.real(sym_prop[1]["TT"])
}

# ================== #
# Plotting (loop over repeated plot code)
wn = np.linspace(-360, 360, 720)
fr = np.linspace(1/24, 2, 48)
wnm, frm = np.meshgrid(wn, fr)

fr_ana, wn_ana = th.genDispersionCurves()
e_cond = np.squeeze(np.where(wn_ana[3, 0, :] <= 0))

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

def plot_growth_rate(ax, growth_rate, wn_ana, fr_ana, e_cond, max, min, color_range):
    c = ax.contourf(wnm, frm, growth_rate, cmap="RdBu_r", levels=np.linspace(min, max), extend=color_range, norm=TwoSlopeNorm(vmin=min, vmax=max, vcenter=0))
    for i in range(3):
        ax.plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=1, linestyle=":")
        ax.plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=1, linestyle=":")
        ax.plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=1, linestyle=":")
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 0.5)
    plt.colorbar(c, ax=ax)

plot_growth_rate(axes[0], growth_rate["pc1"], wn_ana, fr_ana, e_cond, 6, -6, "both")
plot_growth_rate(axes[1], growth_rate["pc2"], wn_ana, fr_ana, e_cond, 6, -6, "both")

plt.show()
