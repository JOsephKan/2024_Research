# This program is to compute power spectrum in PCs
# import package
import sys
import numpy as np
import netCDF4 as nc

from matplotlib import pyplot as plt 

def plot_setting():
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.titlesize"] = 32

def main():

    # import self-defined function
    plot_setting()

    sys.path.append("/home/b11209013/Package/")
    import Theory as th
    import DataProcess as dp
    import SignalProcess as sp
    
    data = {}
    # this file is temperature principal components in boreal winter from 2000~2021

    tname = "/work/b11209013/MPAS/CNTL/merged_data/tropic_data/t/tPC.nc"

    with nc.Dataset(tname, "r") as f:
        lat = f.variables["lat"][:]
        lon = f.variables["lon"][:]
        data["tpc1"] = f.variables["tpc1"][:]
        data["tpc2"] = f.variables["tpc2"][:]

    # this file is Q1 principal components in boreal winter from 2000~2021
    q1name = "/work/b11209013/MPAS/CNTL/merged_data/tropic_data/Q1/Q1PC.nc"

    with nc.Dataset(q1name, "r") as f:
        data["q1pc1"] = f.variables["q1pc1"][:]*86400/1004.5
        data["q1pc2"] = f.variables["q1pc2"][:]*86400/1004.5

    ltime, llat, llon = data["q1pc1"].shape

    # ========================= #
    # Format data
    sym_data: dict[str, np.ndarray] = {}
    asy_data: dict[str, np.ndarray] = {}

    fmt = dp.Format(lat)

    for key in data.keys():
        sym_data[key] = fmt.sym(data[key])
        asy_data[key] = fmt.asy(data[key])

    # split data
    sym_split = {}; asy_split = {}

    for key in sym_data.keys():
        sym_split[key] = np.stack(
                [sym_data[key][i*36:i*36+96] for i in range(12)],
                axis=0
                )
        asy_split[key] = np.stack(
                [asy_data[key][i*36:i*36+96] for i in range(12)],
                axis=0
                )

    # ======================== #
    # Propoerties

    ft = sp.Fourier()

    sym: dict[str, np.ndarray] = {
            "pc1":{
                "TT": np.stack(
                    [ft.PowerSpectrum(sym_split["tpc1"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0),
                "QQ": np.stack(
                    [ft.PowerSpectrum(sym_split["q1pc1"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0),
                "QT": np.stack(
                    [ft.CrossSpectrum(sym_split["q1pc1"][i], sym_split["tpc1"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0)
                },
            "pc2":{
                "TT": np.stack(
                    [ft.PowerSpectrum(sym_split["tpc2"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0),
                "QQ": np.stack(
                    [ft.PowerSpectrum(sym_split["q1pc2"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0),
                "QT": np.stack(
                    [ft.CrossSpectrum(sym_split["q1pc2"][i], sym_split["tpc2"][i]) for i in range(12)],
                    axis=0
                    ).mean(axis=0)
                }
            }

    # ========================= #
    # compute variables
    sym_var: dict[str, np.ndarray] = {
            "sigma":{
                "pc1": 2*sym["pc1"]["QT"].real/sym["pc1"]["TT"],
                "pc2": 2*sym["pc2"]["QT"].real/sym["pc2"]["TT"]
                },
            "phi": {
                "pc1": np.arctan(np.imag(sym["pc1"]["QT"]), np.real(sym["pc1"]["QT"])),
                "pc2": np.arctan(np.imag(sym["pc2"]["QT"]), np.real(sym["pc2"]["QT"])),
                },
            "Coh": {
                "pc1": (np.real(sym["pc1"]["QT"])**2 + np.imag(sym["pc1"]["QT"])**2) / (sym["pc1"]["QQ"] * sym["pc1"]["TT"]),
                "pc2": (np.real(sym["pc2"]["QT"])**2 + np.imag(sym["pc2"]["QT"])**2) / (sym["pc2"]["QQ"] * sym["pc2"]["TT"]),
                }
            } 

    # Plot out the diagram of growth rate
    wn = np.linspace(-360, 360, 720)
    fr = np.linspace(1/96, 2, 48)

    wnm, frm = np.meshgrid(wn, fr)

    kel12 = th.Wave(wn, 12)
    kel25 = th.Wave(wn, 25)
    kel50 = th.Wave(wn, 50)

    fig, ax = plt.subplots(1, 2, figsize=(17, 8), sharey="row")

    # cf = ax[0].contourf(wnm, frm, sym_var["sigma"]["pc1"], levels=np.linspace(-1.5, 1.5, 31), extend="both", cmap="coolwarm")
    # #ax[0].quiver(wnm, frm, np.cos(sym_var["phi"]["pc1"]), np.sin(sym_var["phi"]["pc1"]), scale=50)
    # ax[0].plot(wn, kel12.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[0].plot(wn, kel25.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[0].plot(wn, kel50.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[0].set_xticks(np.linspace(-14, 14, 8))
    # ax[0].set_yticks(np.linspace(0, 0.8, 9))
    # ax[0].set_xlim(-15, 15)
    # ax[0].set_ylim(0, 0.8)
    # ax[0].set_xlabel("Zonal Wavenumber")
    # ax[0].set_ylabel("Frequency")
    # ax[0].set_title("PC 1")
    # 
    # cf = ax[1].contourf(wnm, frm, sym_var["sigma"]["pc2"], levels=np.linspace(-1.5, 1.5, 31), extend="both", cmap="coolwarm")
    # #ax[1].quiver(wnm, frm, np.cos(sym_var["phi"]["pc1"]), np.sin(sym_var["phi"]["pc1"]), scale=50)
    # ax[1].plot(wn, kel12.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[1].plot(wn, kel25.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[1].plot(wn, kel50.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    # ax[1].set_xticks(np.linspace(-14, 14, 8))
    # ax[1].set_yticks(np.linspace(0, 0.8, 9))
    # ax[1].set_xlim(-15, 15)
    # ax[1].set_ylim(0, 0.8)
    # ax[1].set_xlabel("Zonal Wavenumber")
    # ax[1].set_title("PC 2")
# 
    # plt.suptitle("EAPE Growth Rate (CNTL)")
# 
    # fig.colorbar(cf, ax=ax, orientation="horizontal", aspect=30, shrink=0.35, label=r"$\sigma$ [1/day]")
    # plt.savefig("/home/b11209013/2024_Research/MPAS_ana/CNTL/image/GrowthRate.png", dpi=500)
    # plt.show()
    plt.close()

    fig= plt.figure(figsize=(8, 6))
    cf = plt.contourf(wnm, frm, sym_var["sigma"]["pc2"], levels=np.linspace(-1.5, 1.5, 31), extend="both", cmap="coolwarm")
    plt.plot(wn, kel12.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    plt.plot(wn, kel25.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    plt.plot(wn, kel50.Kelvin(), color="black", linestyle="--", linewidth=0.7)
    plt.xticks(np.linspace(-14, 14, 8))
    plt.yticks(np.linspace(0, 0.8, 9))
    plt.xlim(-15, 15)
    plt.ylim(0, 0.8)
    plt.xlabel("Zonal Wavenumber")
    plt.ylabel("Frequency")
    plt.title("CNTL")
    plt.colorbar(cf, orientation="vertical", label=r"$\sigma$ [1/day]")
    plt.savefig("/home/b11209013/2024_Research/MPAS_ana/CNTL/image/GrowthRate_EOF2.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    main()
