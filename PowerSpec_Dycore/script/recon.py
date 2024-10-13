# This program is to reconstruct the signal of the signal of Kelvin waves
# Basic Settring 
# # import package
import sys
import h5py
import numpy as np

from cartopy import crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter 

from matplotlib import colors as cm
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp

# ================== #
# functions
# # plot setting
def plot_setting():
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.family"] = "serif"

# Gaussian filter
def background(ave, cr=10):
#    ave = (sym + asy) / 2

    low = dp.GaussianFilter(ave[:cr], 40)
    high = dp.GaussianFilter(ave[cr:], 10)

    space_filt = np.concatenate([low, high], axis=0).T

    bg = dp.GaussianFilter(space_filt, 10).T

    return bg


# ================== #
# main function
def main():
    # load data
    path:str = "/home/b11209013/2024_Research/PowerSpec_Dycore/data/pr50/"

    # # range latitude
    lat = np.linspace(-90, 90, 64)
    
    lat_cond = np.where((lat>=-30)&(lat<=30))[0]
    lat = lat[lat_cond]

    # # prec data
    with h5py.File(path+"prec_pr50.dat") as f:
        prec = f["prec"][:, lat_cond]

    prec -= prec.mean(axis=0)
    prec *= 86400

    # # u-wind
    with h5py.File(path+"u_pr50.dat") as f:
        u = f["u"][:, 6, lat_cond]

    u -= u.mean(axis=0)

    # # v-wind
    with h5py.File(path+"v_pr50.dat") as f:
        v = f["v"][:, 6, lat_cond]

    v -= v.mean(axis=0)

    # ================= #
    # Reconstruct Kelvin waves
    # # FFT
    # # # precpitation
    prec_fft = np.array([
        np.fft.fftshift(np.fft.fft2(prec[:, i, :]))[:, ::-1]
        for i in range(len(lat_cond))
        ])

    # # # u-wind
    u_fft = np.array([
        np.fft.fftshift(np.fft.fft2(u[:, i, :]))[:, ::-1]
        for i in range(len(lat_cond))
        ])

    # # # v-wind
    v_fft = np.array([
        np.fft.fftshift(np.fft.fft2(v[:, i, :]))[:, ::-1]
        for i in range(len(lat_cond))
        ])

    # ======================== #
    # filt with Gaussian filter
    for i in range(len(lat)):
        prec_bg = background()


    # # dimension for bandpass filter
    wn = np.linspace(-64, 64, 128)
    fr = np.linspace(-2, 2, 78000)

    wnm, frm = np.meshgrid(wn, fr)
    
    ed8 = th.Wave(wnm, 8)
    ed90 = th.Wave(wnm, 90)

    cond = np.logical_or.reduce([
        wnm < 1, wnm > 14,
        frm < 1/20, frm > 1/2.5,
        frm < ed8.Kelvin(), frm > ed90.Kelvin()
        ])

    # # bandpass filter and inverse FFT
    # # # IFFT
    prec_ifft = []
    u_ifft = []
    v_ifft = []

    for i in range(len(lat_cond)):
        prec_filted = np.where(cond == True, 0, prec_fft[i]*2)
        prec_ifft.append(np.fft.ifft2(np.fft.ifftshift(prec_filted)))

        u_filted = np.where(cond == True, 0, u_fft[i]*2)
        u_ifft.append(np.fft.ifft2(np.fft.ifftshift(u_filted)))

        v_filted = np.where(cond == True, 0, v_fft[i]*2)
        v_ifft.append(np.fft.ifft2(np.fft.ifftshift(v_filted)))

    # # # reconstruct
    prec_recon = np.real(np.stack(prec_ifft, axis=1))
    u_recon = np.real(np.stack(u_ifft, axis=1))
    v_recon = np.real(np.stack(v_ifft, axis=1))

    prec_var = np.var(prec_recon, axis=0)
    
    max_v = np.unravel_index(np.argmax(prec_var), prec_var.shape)

    ref = prec_recon[:, max_v[0], max_v[1]]

    pos_95 = np.where(ref >= ref.mean()+1.96*ref.std())[0]

    # ================ #
    prec_sel = prec_recon[pos_95].mean(axis=0)
    u_sel = u_recon[pos_95].mean(axis=0)
    v_sel = v_recon[pos_95].mean(axis=0)

    # plot out the map
    lon = np.linspace(0, 359, 128)
    lonm, latm = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(15, 7))
    c = plt.contourf(
            lonm, latm, prec_sel,
            cmap="RdBu_r",
            extend="both",
            levels=np.linspace(-3, 3, 13)
            )
    q = plt.quiver(lonm[:, ::3], latm[:, ::3], u_sel[:, ::3], v_sel[:, ::3], width=0.001, scale=20)
    plt.ylim(-30, 30)
    plt.colorbar(c)
    plt.show()
    
    print(q.scale)
# ===================== #
# execution section
if __name__ == "__main__":
    main()
