# This program is to compute power spectrum of total experiment sets
# import packages
import sys
import numpy as np
import netCDF4 as nc

from matplotlib import pyplot as plt

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# ============================= #
# functions


# # plot setting
def plot_setting():
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.titlesize"] = 16


# # load data
def load_data(case: str) -> dict[str, np.ndarray]:
    data_dict: dict[str, np.ndarray] = {}

    with nc.Dataset(
            f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/Q1/Q1.nc"
            ) as f:
        data_dict["lon"] = f.variables["lon"][:]
        data_dict["lat"] = f.variables["lat"][:]
        data_dict["lev"] = f.variables["lev"][:] * 100
        data_dict["Q1"] = f.variables["Q1"][:]

    return data_dict


# # vertical integral
def vertical_integral(data: np.ndarray, lev: np.ndarray) -> np.ndarray:
    data_ave: np.ndarray = (data[:, 1:] + data[:, :-1]) / 2
    data_vint: np.ndarray = (
            -np.sum(data_ave * np.diff(lev)[None, :, None, None], axis=1)
            * 86400
            / 9.8
            / 2.5e6
            )

    return data_vint


# ============================= #


def main():
    plot_setting()

    # load data
    path: str = "/work/b11209013/MPAS/"
    case_list: list[str] = ["CNTL", "NCRF", "NSC", "SD15"]

    data: dict[str, dict[str, np.ndarray]] = {}
    for case in case_list:
        data[case] = load_data(case)

    # ========================= #

    # vertical integral
    data_vint: dict[str, np.ndarray] = {}

    for case in case_list:
        data_vint[case] = vertical_integral(data[case]["Q1"], data[case]["lev"])
        print(data_vint[case].shape)
    # ========================= #
    # Process data
    # # Format data
    data_sym: dict[str, np.ndarray] = {}
    data_asy: dict[str, np.ndarray] = {}

    for case in case_list:
        fmt = dp.Format(data[case]["lat"])
        data_sym[case] = fmt.sym(data_vint[case])
        data_asy[case] = fmt.asy(data_vint[case])

    # # Split data
    data_sym_split: dict[str, np.ndarray] = {}
    data_asy_split: dict[str, np.ndarray] = {}

    for case in case_list:
        sec: int = (data_sym[case].shape[0] - 96) // 36 + 1
        data_sym_split[case] = np.stack(
                [data_sym[case][i * 36: i * 36 + 96] for i in range(sec)], axis=0
                )
        data_asy_split[case] = np.stack(
                [data_asy[case][i * 36: i * 36 + 96] for i in range(sec)], axis=0
                )

    # ========================= #
    # compute power spectrum
    fourier = sp.Fourier()
    data_sym_ps: dict[str, np.ndarray] = {}
    data_asy_ps: dict[str, np.ndarray] = {}

    for case in case_list:
        sec = data_sym_split[case].shape[0]
        data_sym_ps[case] = np.stack(
                [fourier.PowerSpectrum(data_sym_split[case][i]) for i in range(sec)], axis=0
                ).mean(axis=0)
        data_asy_ps[case] = np.stack(
                [fourier.PowerSpectrum(data_asy_split[case][i]) for i in range(sec)], axis=0
                ).mean(axis=0)

    # ========================= #
    # compute ratio between real data and background
    peak: dict[str, np.ndarray] = {}

    ave: np.ndarray = (data_sym_ps["CNTL"] + data_asy_ps["CNTL"]) / 2

    low: np.ndarray = dp.GaussianFilter(ave[:15], 40)
    high: np.ndarray = dp.GaussianFilter(ave[15:], 10)

    sf: np.ndarray = np.concatenate([low, high], axis=0).T
    bg: np.ndarray = dp.GaussianFilter(sf, 10).T

    for case in case_list:
        peak[case] = np.divide(data_sym_ps[case], bg)

    # ========================= #
    # visulization of power spectrum
    fig, ax = plt.subplots(2, 2, figsize=(17, 13), sharex=True, sharey=True)

    # # coordinate
    wn: np.ndarray = np.linspace(-360, 360, 720)
    fr: np.ndarray = np.linspace(1 / 96, 2, 48)
    wnm, frm = np.meshgrid(wn, fr)

    ed12 = th.Wave(wn, 12)
    ed25 = th.Wave(wn, 25)
    ed50 = th.Wave(wn, 50)

    # # figure
    c = ax[0, 0].contourf(
            wnm, frm, peak["CNTL"], levels=np.linspace(1, 10, 10), cmap="RdPu", extend="max"
            )
    ax[0, 0].plot(wn, ed12.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 0].plot(wn, ed25.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 0].plot(wn, ed50.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 0].set_yticks(np.linspace(0, 0.8, 9))
    ax[0, 0].set_xlim(-15, 15)
    ax[0, 0].set_ylim(0, 0.8)
    ax[0, 0].set_ylabel("Frequency")
    ax[0, 0].set_title("CNTL")

    c = ax[0, 1].contourf(
            wnm, frm, peak["NCRF"], levels=np.linspace(1, 10, 10), cmap="RdPu", extend="max"
            )
    ax[0, 1].plot(wn, ed12.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 1].plot(wn, ed25.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 1].plot(wn, ed50.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[0, 1].set_xlim(-15, 15)
    ax[0, 1].set_ylim(0, 0.8)
    ax[0, 1].set_title("NCRF")

    c = ax[1, 0].contourf(
            wnm, frm, peak["NSC"], levels=np.linspace(1, 10, 10), cmap="RdPu", extend="max"
            )
    ax[1, 0].plot(wn, ed12.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 0].plot(wn, ed25.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 0].plot(wn, ed50.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 0].set_xticks(np.linspace(-14, 14, 8))
    ax[1, 0].set_yticks(np.linspace(0, 0.8, 9))
    ax[1, 0].set_xlim(-15, 15)
    ax[1, 0].set_ylim(0, 0.8)
    ax[1, 0].set_xlabel("Zonal Wavenumber")
    ax[1, 0].set_ylabel("Frequency")
    ax[1, 0].set_title("NSC")

    c = ax[1, 1].contourf(
            wnm, frm, peak["SD15"], levels=np.linspace(1, 10, 10), cmap="RdPu", extend="max"
            )
    ax[1, 1].plot(wn, ed12.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 1].plot(wn, ed25.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 1].plot(wn, ed50.Kelvin(), color="black",
            linestyle="--", linewidth=0.5)
    ax[1, 1].set_xticks(np.linspace(-14, 14, 8))
    ax[1, 1].set_xlim(-15, 15)
    ax[1, 1].set_ylim(0, 0.8)
    ax[1, 1].set_xlabel("Zonal Wavenumber")
    ax[1, 1].set_title("SD15")

    plt.suptitle(r"Peak of Symmetric Components of Q1 ($15^\circ S \sim 15^\circ N$)")
    plt.colorbar(c, ax=ax, orientation="horizontal", shrink=0.6, aspect=30)

    plt.savefig("/home/b11209013/MPAS_ana/image/power_spec.png", dpi=500)
    plt.show()


# ============================= #
if __name__ == "__main__":
    main()
