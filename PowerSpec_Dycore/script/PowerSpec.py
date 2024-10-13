# This program is to compute power spectrum of PCs of different variables
# import package
import sys
import h5py
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
    lat = np.linspace(-90, 90, 64)

    cond = np.where((lat>=-15)&(lat<=15))[0]

    lat = lat[cond]

    with h5py.File("/data92/PeterChang/back_to_master1220/Moist_Dycore/IdealizeSpetral.jl/exp/HSt42/6hourly_uv_prime_EMF/PR50/prec/PR50_500_20000day_6hourly_prec.dat") as f:
        prec = f["prec"][:, cond, :]

    prec = prec - prec.mean()

    # ============================== #
    
    # Split data
    
    split = np.array([prec[i*36:i*36+96] for i in range(2165)])

    # ============================== #
    
    # Format data

    fmt = dp.Format(lat)
    
    sym_split = np.array([fmt.sym(split[i]) for i in range(split.shape[0])])
    asy_split = np.array([fmt.asy(split[i]) for i in range(split.shape[0])])

    # ============================== #
    
    # Power Spectrum
   
    ft = sp.Fourier()
    
    sym_ps = np.array([ft.PowerSpectrum(sym_split[i]) for i in range(2165)]).mean(axis=0)
    asy_ps = np.array([ft.PowerSpectrum(asy_split[i]) for i in range(2165)]).mean(axis=0)

    # =============================== #
    
    # peak computation
    
    bg = background(sym_ps, asy_ps, 15)
    sym_peak = sym_ps / bg 
    
    # =============================== #
    
    # Plot Power Spectrum
    
    # ## dimension
    wn = np.linspace(-64, 64, 128)
    fr = np.linspace(1 / 96, 2, 48)
    
    wnm, frm = np.meshgrid(wn, fr)
 
    Kelvin8 = th.Wave(wn, 8)
    Kelvin25 = th.Wave(wn, 25)
    Kelvin90 = th.Wave(wn, 90)
    
    # ## Plot out the diagram
  
    plt.contourf(
        wnm,
        frm,
        sym_peak,
        cmap="Reds",
        levels=np.linspace(1, 6, 36),
        extend="max",
    )
    plt.plot(
        wn, Kelvin8.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    plt.plot(
        wn, Kelvin25.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    plt.plot(
        wn, Kelvin90.Kelvin(), color="black", linestyle=":", linewidth=0.7
    )
    plt.text(7, 0.45, "90 m", fontsize=10)
    plt.text(13, 0.43, "25 m", fontsize=10)
    plt.text(13, 0.23, "8 m", fontsize=10)
    plt.axvline(0, color="black", linewidth=0.5, linestyle=":")
    plt.xticks(np.linspace(-14, 14, 8))
    plt.yticks(np.linspace(0, 0.5, 11))
    plt.xlim(-15, 15)
    plt.ylim(0, 1 / 2)
    plt.xlabel("Zonal Wavenumber")
    plt.title("PR 50")
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    main()
