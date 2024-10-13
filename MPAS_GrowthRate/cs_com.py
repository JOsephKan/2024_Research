# This program is to compute the cross spectrum of the MPAS experiments
# import package
import sys
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# =============== # 
# load data
case = 'CNTL'
var_list = ["q1pc1", "q1pc2", "tpc1", "tpc2"]

path = "/work/b11209013/MPAS/PC/"

data_dict = {}

with open(f"{path}{case}/pc_q1eof.pkl", "rb") as f:
    data = pkl.load(f)
    lon = data["lon"]
    lat = data["lat"]
    time = data["time"]

lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
lat = lat[lat_lim]


with open(f"{path}{case}/pc_q1eof.pkl", "rb") as f:
    data = pkl.load(f)
    data_dict["q1pc1"] = data["q1pc"][0][:, lat_lim, :]
    data_dict["q1pc2"] = data["q1pc"][1][:, lat_lim, :]
    data_dict["tpc1"] = data["tpc"][0][:, lat_lim, :]
    data_dict["tpc2"] = data["tpc"][1][:, lat_lim, :]

data_dict = {key: data_dict[key] - data_dict[key].mean() for key in data_dict}

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
        ft.PowerSpectrum(sym_split["tpc1"][i]).real
        for i in range(4)
    ]).mean(axis=0),
    "pc2": np.array([
        ft.PowerSpectrum(sym_split["tpc2"][i]).real
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

cf_1 = ax[0].contourf(wnm, frm, (sigma["pc1"]), cmap="RdBu_r", levels=np.linspace(-10, 10), extend="both")
ax[0].plot(wn_ana[4, 0, :], fr_ana[4, 0, :], "k")
ax[0].plot(wn_ana[4, 1, :], fr_ana[4, 1, :], "k")
ax[0].plot(wn_ana[4, 2, :], fr_ana[4, 2, :], "k")

ax[0].set_xlim(-15, 15)
ax[0].set_ylim(0, 1/2)
plt.colorbar(cf_1, ax=ax[0])

cf_2 = ax[1].contourf(wnm, frm, (sigma["pc2"]), cmap="RdBu_r", levels=np.linspace(-10, 10), extend="both")
ax[1].plot(wn_ana[4, 0, :], fr_ana[4, 0, :], "k")
ax[1].plot(wn_ana[4, 1, :], fr_ana[4, 1, :], "k")
ax[1].plot(wn_ana[4, 2, :], fr_ana[4, 2, :], "k")

ax[1].set_xlim(-15, 15)
ax[1].set_ylim(0, 1/2)
plt.colorbar(cf_2, ax=ax[1])

plt.show()
