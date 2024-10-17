# This program is to compute the cross spectrum of the MPAS experiments
# import package
import sys
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# =============== # 
# load data
case = 'NSC'
var_list = ["q1pc1", "q1pc2", "tpc1", "tpc2"]

fname = f"/home/b11209013/2024_Research/MPAS/GrowthRate/PCA_file/{case}_PC.nc"

data_dict = {}

with nc.Dataset(fname, 'r') as data:
    lon  = data["lon"][:]
    lat  = data["lat"][:]
    time = data["time"][:] 

    for var in var_list:
        data_dict[var] = data[var][:]

data_dict = {
    key: data_dict[key] - np.mean(data_dict[key], axis=0, keepdims=True)
    for key in var_list
}

# ================= #   
# process data
fmt = dp.Format(lat)

sym = {
    var: fmt.sym(data_dict[var])
    for var in var_list
}
del data_dict

# process data
sym_split = {
    var: np.array(
        [sym[var][i*40:i*40+240] for i in range(4)]
        )
    for var in var_list
}


del sym

# =============== # 
ft = sp.Fourier()

TT = {
    "pc1": np.array([
        ft.CrossSpectrum(sym_split["tpc1"][i], sym_split["tpc1"][i]).real
        for i in range(4)
    ]).mean(axis=0),
    "pc2": np.array([
        ft.CrossSpectrum(sym_split["tpc2"][i], sym_split["tpc2"][i]).real
        for i in range(4)
    ]).mean(axis=0),
}

QT = {
    "pc1": np.array([
        ft.CrossSpectrum(sym_split["q1pc1"][i], sym_split["tpc1"][i])
        for i in range(4)
    ]).mean(axis=0),
    "pc2": np.array([
        ft.CrossSpectrum(sym_split["q1pc2"][i], sym_split["tpc2"][i])
        for i in range(4)
    ]).mean(axis=0),
}


del sym_split

sigma = {
    "pc1": 2*np.real(QT["pc1"])/TT["pc1"],
    "pc2": 2*np.real(QT["pc2"])/TT["pc2"]
}

# ============== # 
wn = np.linspace(-359, 360, 720)
fr = np.linspace(1/60, 2, 120)

wnm, frm = np.meshgrid(wn, fr)

fr_ana, wn_ana = th.genDispersionCurves(Ahe=[90, 25, 8])
e_cond = np.squeeze(np.where(wn_ana[3, 0, :] <= 0))

fig, ax = plt.subplots(1, 2, figsize=(17, 7))

cf_1 = ax[0].contourf(wnm, frm, (sigma["pc1"]), cmap="coolwarm", levels=np.linspace(-10, 10, 11), extend="both")
ax[0].plot(wn_ana[4, 0, :], fr_ana[4, 0, :], "k")
ax[0].plot(wn_ana[4, 1, :], fr_ana[4, 1, :], "k")
ax[0].plot(wn_ana[4, 2, :], fr_ana[4, 2, :], "k")

ax[0].set_xticks(np.linspace(-14, 14, 8))
ax[0].set_yticks(np.linspace(0, 0.5, 6))
ax[0].set_xlim(-15, 15)
ax[0].set_ylim(0, 1/2)
ax[0].set_xlabel("Zonal Wavenumber")
ax[0].set_ylabel("Frequency (CPD)")
ax[0].set_title("PC1")

plt.colorbar(cf_1, ax=ax[0])

cf_2 = ax[1].contourf(wnm, frm, (sigma["pc2"]), cmap="coolwarm", levels=np.linspace(-5, 5, 11), extend="both")
ax[1].plot(wn_ana[4, 0, :], fr_ana[4, 0, :], "k")
ax[1].plot(wn_ana[4, 1, :], fr_ana[4, 1, :], "k")
ax[1].plot(wn_ana[4, 2, :], fr_ana[4, 2, :], "k")

ax[1].set_xticks(np.linspace(-14, 14, 8))
ax[1].set_yticks(np.linspace(0, 0.5, 6))
ax[1].set_xlim(-15, 15)
ax[1].set_ylim(0, 1/2)
ax[1].set_xlabel("Zonal Wavenumber")
ax[1].set_ylabel("Frequency (CPD)")
ax[1].set_title("PC2")
ax[1].set_xlim(-15, 15)
ax[1].set_ylim(0, 1/2)
plt.colorbar(cf_2, ax=ax[1])

plt.savefig(f"/home/b11209013/2024_Research/MPAS/GrowthRate/image/{case}_gr.png", dpi=500)
plt.show()
plt.close()
