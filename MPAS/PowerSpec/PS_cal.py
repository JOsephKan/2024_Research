# This program is to compute the power spectrum of the Q1 principal component
# %% section 1
# import package
## 1. standard package
import sys
import numpy as np
import joblib as jl

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

    return ps

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
    file = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.joblib')

    lat = file['lat']
    q1_arr = file['pc']['q1']
    
    data['pc1'][case] = q1_arr[0]
    data['pc2'][case] = q1_arr[1]

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

plt.contourf(wnm_v, frm_v, np.log(np.fft.fftshift(sym_ps['pc1']['CNTL'])), cmap='jet')
plt.xlim(-15, 15)
plt.ylim(0, 0.5)
plt.colorbar()
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

# %%
# 3. compute the background
bg: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: backgroud(sym_ps[pc][case], asy_ps[pc][case])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

plt.contourf(wnm_v, frm_v, np.log(np.fft.fftshift(bg['pc1']['CNTL'])), cmap='jet')
plt.xlim(-15, 15)
plt.ylim(0, 0.5)
plt.colorbar()

sym_peak: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: sym_ps[pc][case] / bg[pc][case]
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}


# %% section 6
# plot the power spectrum
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["font.family"] = "serif"

plt.contourf(wnm_v, frm_v, np.fft.fftshift(sym_peak['pc1']['CNTL']), cmap='jet')
#plt.xlim(-15, 15)
#plt.ylim(0, 0.5)
# %%
