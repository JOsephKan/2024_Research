# This program is to compute the power spectrum of the Q1 principal component
# %% section 1
# import package
## 1. standard package
import sys
import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt

## 2. user-defined package
sys.path.append('/home/b11209013/Package')
import Theory as th        #type: ignore
import DataProcess as dp   #type: ignore
import SignalProcess as sp #type: ignore

# %% section 2
# 1. Power Spectrum
def PowerSpectrum(
    data: np.ndarray,
) -> np.ndarray:
    ltime, llon = data.shape
    
    wn = np.fft.fftfreq(llon, d=1/llon)
    fr = np.fft.fftfreq(ltime, d=1/4)

    wnm, frm = np.meshgrid(wn, fr)
    
    mask = np.where(frm>0, 2, 0)
    
    fft = np.array([np.fft.fft(data[i]) for i in range(ltime)])
    fft = np.array([np.fft.ifft(fft[:, i]) for i in range(llon)]).T
    
    ps = ((fft*np.conj(fft)) / (ltime*llon))

    ps = ps*mask

    return ps.real

# 2. Background
def backgroud(sym, asy):
    ps_ave = (sym + asy) / 2

    low = dp.GaussianFilter(ps_ave[:40], 40)
    high = dp.GaussianFilter(ps_ave[40:], 10)

    s_f = np.concatenate([low, high], axis=0).T

    bg = dp.GaussianFilter(s_f, 10).T
    
    return bg

# %% section 3
# load the data
cases: list[str] = ['CNTL', 'NCRF', 'NSC']

data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {},
    'pc2': {},
}

for case in cases:
    with nc.Dataset(f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.nc') as f:
        lat = f['lat'][:]

        data['pc1'][case] = f['q1'][0].transpose(2, 1, 0)
        data['pc2'][case] = f['q1'][1].transpose(2, 1, 0)


print(data['pc1']['CNTL'].shape)

ltime, llat, llon = data['pc1']['CNTL'].shape


# %% section 4
# processing data
# 1. symmetric data
fmt = dp.Format(lat)

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: fmt.sym(data[pc][case])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

asy: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: fmt.asy(data[pc][case])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

# 2. Windowing
hann = np.hanning(120)[:, None]

sym_window: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: np.array([
            sym[pc][case][i*60:i*60+120] * hann
            for i in range(5)
        ])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

asy_window: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: np.array([
            asy[pc][case][i*60:i*60+120] * hann
            for i in range(5)
        ])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

# %% section 5
# 1. setting wavenumber and frequency
wn = np.fft.fftfreq(llon, d=1/llon)
fr = np.fft.fftfreq(120, d=1/4)

wnm, frm = np.meshgrid(wn, fr)

wnm_v = np.fft.fftshift(wnm) # used for visualization
frm_v = np.fft.fftshift(frm) # used for visualization

# 2. compute the power spectrum
sym_ps: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: np.array([
            PowerSpectrum(sym_window[pc][case][i])
            for i in range(5)
        ]).mean(axis=0)
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

asy_ps: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: np.array([
            PowerSpectrum(asy_window[pc][case][i])
            for i in range(5)
        ]).mean(axis=0)
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}


# filtered out
def pos_cond(cond, data):
    data_new = np.fft.fftshift(data)[frm_v>0].reshape(-1, llon)

    return data_new

sym_ps = {
    pc:{
        case: pos_cond(frm_v>0, sym_ps[pc][case])
        for case in cases
    }
    for pc in ["pc1", "pc2"]
}

asy_ps = {
    pc:{
        case: pos_cond(frm_v>0, asy_ps[pc][case])
        for case in cases
    }
    for pc in ["pc1", "pc2"]
}

# %%
# 3. compute the background
bg: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: backgroud(sym_ps[pc][case], asy_ps[pc][case])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

peak: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: sym_ps[pc][case] / bg[pc]["CNTL"]
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}


# %% section 6
# plot the power spectrum
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["font.size"] = 10
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["font.family"] = "serif"

wnm = wnm_v[frm_v>0].reshape(-1, llon)
frm = frm_v[frm_v>0].reshape(-1, llon)


fr_ana, wn_ana = th.genDispersionCurves()
e_cond = np.where(wn_ana[3, 0, :] <=0)

fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

c1 = ax[0, 0].contourf(
    wnm, frm, (peak["pc1"]["CNTL"]),
    levels=np.linspace(1, 10),
    cmap="terrain_r", extend="max"
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
    cmap="terrain_r", extend="max"
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
    cmap="terrain_r", extend="max"
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
plt.savefig("/home/b11209013/2024_Research/MPAS/PowerSpec/MPAS_pc1.png", dpi=300)
plt.show()
plt.close()


fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

c1 = ax[0, 0].contourf(
    wnm, frm, (peak["pc2"]["CNTL"]),
    levels=np.linspace(1, 10),
    cmap="terrain_r", extend="max"
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
    cmap="terrain_r", extend="max"
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
    cmap="terrain_r", extend="max"
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
plt.savefig("/home/b11209013/2024_Research/MPAS/PowerSpec/MPAS_pc2.png", dpi=300)
plt.show()
plt.close()
