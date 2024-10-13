# This program is to compute power spectrum of the CNTL experiment
# import package
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# =============== # 
# function
def load_data(path):
    with open(f"{path}q1pc_q1eof.pkl", "rb") as f:
        pc_dict = pkl.load(f)
    
    return pc_dict

def backgroud(sym, asy):
    ps_ave = (sym + asy) / 2

    low = dp.GaussianFilter(ps_ave[:40], 40)
    high = dp.GaussianFilter(ps_ave[40:], 10)

    s_f = np.concatenate([low, high], axis=0).T

    bg = dp.GaussianFilter(s_f, 10).T
    
    return bg


# =============== # 
# load data
case = ["CNTL", "NCRF", "NSC"]
path = [f"/work/b11209013/MPAS/PC/{c}/" for c in case]

pc_dict = {key: load_data(path[i]) for i, key in enumerate(case)}

lat  = pc_dict["CNTL"]["lat"]
lon  = pc_dict["CNTL"]["lon"]
time = pc_dict["CNTL"]["time"]

lat_lim = np.where((lat >= -5) & (lat <= 5))[0]

lat = lat[lat_lim]

pc1 = {key: pc_dict[key]["pc"][0][:, lat_lim, :] for key in case}
pc1 = {key: pc1[key]-pc1[key].mean(axis=0) for key in pc1.keys()}


pc2 = {key: pc_dict[key]["pc"][1][:, lat_lim, :] for key in case}
pc2 = {key: pc2[key]-pc2[key].mean(axis=0) for key in pc2.keys()}

# ================== #
# format data
fmt = dp.Format(lat)

sym = {
    "pc1": {
        key: fmt.sym(pc1[key]) for key in case
    },
    "pc2": {
        key: fmt.sym(pc2[key]) for key in case
    }
}
asy = {
    "pc1": {
        key: fmt.asy(pc1[key]) for key in pc1.keys()
    },
    "pc2": {
        key: fmt.asy(pc2[key]) for key in pc2.keys()
    }
}

# split data
sym_split = {
    pc: {
        c: np.array([
            sym[pc][c][i*40:i*40+240, :]
            for i in range(4)
        ])
        for c in case
    }
    for pc in sym.keys()
}
asy_split = {
    pc: {
        c: np.array([
            asy[pc][c][i*40:i*40+240, :]
            for i in range(4)
        ])
        for c in case
    }
    for pc in asy.keys()
}

# ================== #
# compute power spectrum

fourier = sp.Fourier()

sym_ps = {
    pc: {
        c: np.array([
            fourier.PowerSpectrum(sym_split[pc][c][i])
            for i in range(4)
        ]).mean(axis=0)
        for c in case
    }
    for pc in sym_split.keys()
}

asy_ps = {
    pc: {
        c: np.array([
            fourier.PowerSpectrum(asy_split[pc][c][i])
            for i in range(4)
        ]).mean(axis=0)
        for c in case
    }
    for pc in asy_split.keys()
}

# ==================== # 
# compute backgroud
pc1_bg = backgroud(sym_ps["pc1"]["CNTL"], asy_ps["pc1"]["CNTL"])
pc2_bg = backgroud(sym_ps["pc2"]["CNTL"], asy_ps["pc2"]["CNTL"])

peak = {
    "pc1": {
        c: sym_ps["pc1"][c] / pc1_bg
        for c in case
    },
    "pc2": {
        c: sym_ps["pc2"][c] / pc2_bg
        for c in case
    },
}

# ==================== #
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["font.family"] = "serif"

# plot out the power spectrum
wn = np.linspace(-359, 360, 720)
fr = np.linspace(1/60, 2, 120)

fr_ana, wn_ana = th.genDispersionCurves()

e_cond = np.where(wn_ana[3, 0, :] <= 0)

wnm, frm = np.meshgrid(wn, fr)

fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

c1 = ax[0, 0].contourf(
    wnm, frm, (peak["pc1"]["CNTL"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[0, 0].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[0, 0].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[0, 0].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[0, 0].set_xticks(np.linspace(-14, 14, 8))
ax[0, 0].set_yticks(np.linspace(0, 0.5, 6))
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_ylim(0, 0.5)
ax[0, 0].set_ylabel("Frequency (CPD)")
ax[0, 0].set_title("CNTL")
#cb1 = plt.colorbar(c1, ax=ax[0, 0])


c2 = ax[0, 1].contourf(
    wnm, frm, (peak["pc1"]["NCRF"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[0, 1].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[0, 1].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[0, 1].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[0, 1].set_xticks(np.linspace(-14, 14, 8))
ax[0, 1].set_yticks(np.linspace(0, 0.5, 6))
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_ylim(0, 0.5)
ax[0, 1].set_xlabel("Zonal Wavenumber")
# ax[0, 1].set_ylabel("Frequency (CPD)")
ax[0, 1].set_title("NCRF")
# plt.colorbar(c2, ax=ax[0, 1])

c3 = ax[1, 0].contourf(
    wnm, frm, (peak["pc1"]["NSC"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[1, 0].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[1, 0].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[1, 0].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[1, 0].set_xticks(np.linspace(-14, 14, 8))
ax[1, 0].set_yticks(np.linspace(0, 0.5, 6))
ax[1, 0].set_xlim(-15, 15)
ax[1, 0].set_ylim(0, 0.5)
ax[1, 0].set_xlabel("Zonal Wavenumber")
ax[1, 0].set_ylabel("Frequency (CPD)")
ax[1, 0].set_title("NSC")
# plt.colorbar(c3, ax=ax[1, 0])

ax[1, 1].axis("off")

cb = plt.colorbar(c3, ax=ax, label="Normalized Power Spectrum", aspect=40)
cb.set_ticks(np.linspace(1, 10, 10))

plt.suptitle("PC1 (symmetric) Normalized Power Spectra")
plt.savefig("/home/b11209013/2024_Research/MPAS_PowerSpectrum/MPAS_pc1.png", dpi=500)

plt.show()
plt.close()


fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

c1 = ax[0, 0].contourf(
    wnm, frm, (peak["pc2"]["CNTL"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[0, 0].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[0, 0].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[0, 0].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[0, 0].set_xticks(np.linspace(-14, 14, 8))
ax[0, 0].set_yticks(np.linspace(0, 0.5, 6))
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_ylim(0, 0.5)
ax[0, 0].set_ylabel("Frequency (CPD)")
ax[0, 0].set_title("CNTL")
#cb1 = plt.colorbar(c1, ax=ax[0, 0])


c2 = ax[0, 1].contourf(
    wnm, frm, (peak["pc2"]["NCRF"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[0, 1].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[0, 1].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[0, 1].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[0, 1].set_xticks(np.linspace(-14, 14, 8))
ax[0, 1].set_yticks(np.linspace(0, 0.5, 6))
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_ylim(0, 0.5)
ax[0, 1].set_xlabel("Zonal Wavenumber")
# ax[0, 1].set_ylabel("Frequency (CPD)")
ax[0, 1].set_title("NCRF")
# plt.colorbar(c2, ax=ax[0, 1])

c3 = ax[1, 0].contourf(
    wnm, frm, (peak["pc2"]["NSC"]),
    levels=np.linspace(1, 10),
    cmap="BuGn", extend="max"
)
for i in range(3):
    ax[1, 0].plot(wn_ana[3, i, :][e_cond], fr_ana[3, i, :][e_cond], color="black", linewidth=0.5, linestyle=":")
    ax[1, 0].plot(wn_ana[4, i, :], fr_ana[4, i, :], color="black", linewidth=0.5, linestyle=":")
    ax[1, 0].plot(wn_ana[5, i, :], fr_ana[5, i, :], color="black", linewidth=0.5, linestyle=":")
ax[1, 0].set_xticks(np.linspace(-14, 14, 8))
ax[1, 0].set_yticks(np.linspace(0, 0.5, 6))
ax[1, 0].set_xlim(-15, 15)
ax[1, 0].set_ylim(0, 0.5)
ax[1, 0].set_xlabel("Zonal Wavenumber")
ax[1, 0].set_ylabel("Frequency (CPD)")
ax[1, 0].set_title("NSC")
# plt.colorbar(c3, ax=ax[1, 0])

ax[1, 1].axis("off")

cb = plt.colorbar(c3, ax=ax, label="Normalized Power Spectrum", aspect=40)
cb.set_ticks(np.linspace(1, 10, 10))

plt.suptitle("PC2 (symmetric) Normalized Power Spectra")
plt.savefig("/home/b11209013/2024_Research/MPAS_PowerSpectrum/MPAS_pc2.png", dpi=500)
plt.show()
plt.close()
