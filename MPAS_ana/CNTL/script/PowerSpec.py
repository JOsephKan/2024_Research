# This program is to compute power spectrum of PCs of different variables
# import package
import sys
import numpy as np
import netCDF4 as nc

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.append("/home/b11209013/Package/")
import Theory as th 
import DataProcess as dp
import SignalProcess as sp

plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["font.family"] = "serif"

# ============================= #

# functions
# Gaussian filter
def background(sym, asy, cr=10):
    ave = (sym + asy) / 2

    low = dp.GaussianFilter(ave[:cr], 40)
    high = dp.GaussianFilter(ave[cr:], 10)

    space_filt = np.concatenate([low, high], axis=0).T

    bg = dp.GaussianFilter(space_filt, 10).T

    return bg


# ============================= #
def main():
    # Load data
    data = {}
    
    path: str = "/work/b11209013/MPAS/CNTL/merged_data/tropic_data/"
    with nc.Dataset(path+"Q1/Q1PC.nc", "r") as f:
        lat = f.variables["lat"][:]
        lon = f.variables["lon"][:]
        data["q1pc1"] = f.variables["q1pc1"][:]
        data["q1pc2"] = f.variables["q1pc2"][:]
    
    with nc.Dataset(path+"t/tPC.nc", "r") as f:
        data["tpc1"] = f.variables["tpc1"][:]
        data["tpc2"] = f.variables["tpc2"][:]
    
    ltime, llat, llon = data["tpc1"].shape
    
    # ============================== #
    
    # Split data
    
    split = {}
    
    for key in data.keys():
        split[key] = np.stack(
            [data[key][i * 36 : i * 36 + 96] for i in range(12)], axis=0
        )
        
    del data
    # ============================== #
    
    # Format data    
    sym_split = {}
    asy_split = {}
    
    fmt = dp.Format(lat)
    
    for key in split.keys():
        sym_split[key] = np.stack([fmt.sym(split[key][i]) for i in range(12)],
                axis=0)
        asy_split[key] = np.stack([fmt.asy(split[key][i]) for i in range(12)],
                axis=0)

    del split
    
    # ============================== #
    
    # Power Spectrum
    
    sym_ps = {}
    asy_ps = {}
    
    ft = sp.Fourier()
    
    for key in sym_split.keys():
        sym_ps[key] = np.stack([ft.PowerSpectrum(sym_split[key][i]) for i in range(12)], axis=0).mean(
            axis=0
        )
        asy_ps[key] = np.stack([ft.PowerSpectrum(asy_split[key][i]) for i in range(12)], axis=0).mean(
            axis=0
        )
    
    del sym_split, asy_split

    # =============================== #
    
    # peak computation
    
    sym_peak = {}
    
    for key in sym_ps.keys():
    
        bg = background(sym_ps[key], asy_ps[key], 15)
        sym_peak[key] = sym_ps[key] / bg
    
    
    # =============================== #
    
    # Plot Power Spectrum
    
    # ## dimension
    wn = np.linspace(-360, 360, 720)
    fr = np.linspace(1 / 96, 2, 48)
    
    wnm, frm = np.meshgrid(wn, fr)
    plt.contourf(wnm, frm, np.log(sym_ps["q1pc1"]))
    plt.xlim(-15, 15)
    plt.ylim(0, 1/2)
    plt.colorbar()
    plt.show()
    Kelvin8 = th.Wave(wn, 8)
    Kelvin25 = th.Wave(wn, 25)
    Kelvin90 = th.Wave(wn, 90)
    
    # ## Plot out the diagram
    
    fig, ax = plt.subplots(2, 2, figsize=(17, 13), sharex="col", sharey="row")
    
    
    c = ax[0, 0].contourf(
        wnm,
        frm,
        sym_peak["q1pc1"].real,
        cmap="Reds",
        levels=np.linspace(1, 6, 36),
        extend="max",
    )
    ax[0, 0].plot(
        wn, Kelvin8.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 0].plot(
        wn, Kelvin25.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 0].plot(
        wn, Kelvin90.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 0].text(7, 0.45, "90 m", fontsize=10)
    ax[0, 0].text(13, 0.43, "25 m", fontsize=10)
    ax[0, 0].text(13, 0.23, "8 m", fontsize=10)
    ax[0, 0].axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax[0, 0].set_yticks(np.linspace(0, 0.5, 11))
    ax[0, 0].set_xlim(-15, 15)
    ax[0, 0].set_ylim(0, 1 / 2)
    ax[0, 0].set_ylabel("Frequency (Cpd)")
    ax[0, 0].set_title("Q1 PC1")
    
    c = ax[0, 1].contourf(
        wnm,
        frm,
        sym_peak["q1pc2"].real,
        cmap="Reds",
        levels=np.linspace(1, 6, 36),
        extend="max",
    )
    ax[0, 1].plot(
        wn, Kelvin8.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 1].plot(
        wn, Kelvin25.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 1].plot(
        wn, Kelvin90.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[0, 1].text(7, 0.45, "90 m", fontsize=10)
    ax[0, 1].text(13, 0.43, "25 m", fontsize=10)
    ax[0, 1].text(13, 0.23, "8 m", fontsize=10)
    ax[0, 1].set_xlim(-15, 15)
    ax[0, 1].axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax[0, 1].set_xlim(-15, 15)
    ax[0, 1].set_ylim(0, 1 / 2)
    ax[0, 1].set_title("Q1 PC2")
    
    c = ax[1, 0].contourf(
        wnm,
        frm,
        sym_peak["tpc1"].real,
        cmap="Reds",
        levels=np.linspace(1, 6, 36),
        extend="max",
    )
    ax[1, 0].plot(
        wn, Kelvin8.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 0].plot(
        wn, Kelvin25.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 0].plot(
        wn, Kelvin90.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 0].text(7, 0.45, "90 m", fontsize=10)
    ax[1, 0].text(13, 0.43, "25 m", fontsize=10)
    ax[1, 0].text(13, 0.23, "8 m", fontsize=10)
    ax[1, 0].axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax[1, 0].set_xticks(np.linspace(-14, 14, 8))
    ax[1, 0].set_yticks(np.linspace(0, 0.5, 11))
    ax[1, 0].set_xlim(-15, 15)
    ax[1, 0].set_ylim(0, 1 / 2)
    ax[1, 0].set_xlabel("Zoanl Wavenumber")
    ax[1, 0].set_ylabel("Frequency (Cpd)")
    ax[1, 0].set_title("T PC1")
    
    c = ax[1, 1].contourf(
        wnm,
        frm,
        sym_peak["tpc2"].real,
        cmap="Reds",
        levels=np.linspace(1, 6, 36),
        extend="max",
    )
    ax[1, 1].plot(
        wn, Kelvin8.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 1].plot(
        wn, Kelvin25.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 1].plot(
        wn, Kelvin90.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    ax[1, 1].text(7, 0.45, "90 m", fontsize=10)
    ax[1, 1].text(13, 0.43, "25 m", fontsize=10)
    ax[1, 1].text(13, 0.23, "8 m", fontsize=10)
    ax[1, 1].axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax[1, 1].set_xticks(np.linspace(-14, 14, 8))
    ax[1, 1].set_yticks(np.linspace(0, 0.5, 11))
    ax[1, 1].set_xlim(-15, 15)
    ax[1, 1].set_ylim(0, 1 / 2)
    ax[1, 1].set_xlabel("Zonal Wavenumber")
    ax[1, 1].set_title("T PC2")
    plt.colorbar(c, ax=ax, orientation="horizontal")
    
    plt.suptitle("Peak of Symmetric Components", fontsize=16)
    #plt.savefig("/home/b11209013/MPAS_ana/CNTL/image/PowerSpec.png", dpi=500)
    
    plt.show()

if __name__ == "__main__":
    main()
