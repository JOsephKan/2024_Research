# This program is to compute the growth rate for heating generated by MPAS model
# %% Section 1: Environment Setup
## 1. Import packages
import os
import sys
import numpy as np
import joblib as jl

from matplotlib import pyplot as plt
## 2. Import self-defined package
sys.path.append('/home/b11209013/Package')
import Theory as th      #type: ignore
import DataProcess as dp #type: ignore

## 3. Accept environment variables
exp: str = 'NSC'

# %% Section 1.5: Functions
## 1. Cross spectrum
def Covariance(data1, data2):
    
    data1 -= data1.mean()
    data2 -= data2.mean()

    fft1 = np.array([np.fft.fft(data1[i]) for i in range(data1.shape[0])])
    fft1 = np.array([np.fft.ifft(fft1[:, i]) for i in range(fft1.shape[1])]).T

    fft2 = np.array([np.fft.fft(data2[i]) for i in range(data2.shape[0])])
    fft2 = np.array([np.fft.ifft(fft2[:, i]) for i in range(fft2.shape[1])]).T

    cs = (fft1*fft2.conj()) / np.prod(data1.shape)

    cs_smooth = np.empty(cs.shape, dtype = complex)
    
    kernel = np.array([1, 2, 1]) / 4
    
    for i in range(cs.shape[0]):
        cs_smooth[i] = np.convolve(cs[i], kernel, mode='same')
    
    for i in range(cs.shape[1]):
        cs_smooth[:, i] = np.convolve(cs_smooth[:, i], kernel, mode='same')
    
    return cs_smooth

## 2. Growth rate
def Growth_Rate(data1, data2):
    
    var = np.array([
        Covariance(data2[i], data2[i]).real
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    cov = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    
    sigma = 2*np.real(cov) / var
    
    return sigma

## 3. Phase
def Phase(data1, data2):
    
    cs = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data1.shape[0])
    ])
    
    phase = np.atan2(cs.imag, cs.real)
    
    return phase

## 4. Coherece Square
def Coherence(data1, data2):
    
    var1 = np.array([
        Covariance(data1[i], data1[i]).real
        for i in range(data1.shape[0])
    ]).mean(axis=0)
    var2 = np.array([
        Covariance(data2[i], data2[i]).real
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    cov  = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data1.shape[0])
    ]).mean(axis=0)
    
    Coh2 = ((cov.real)**2 + (cov.imag)**2) / (var1 * var2)

    return Coh2

# %% Section 2: Load Data
## 1. Load PC data
## path of PC file
path: str = '/work/b11209013/2024_Research/MPAS/PC/'

data = jl.load(path + exp + '_PC.joblib')

## Packed into different dictionaries
### Dimension
dims: dict[str, np.ndarray] ={
    'lat' : data['lat'],
    'lon' : data['lon'],
    'time': data['time'],
} 

### Variables
data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: data['pc'][var][0]
        for var in data['pc'].keys()
    },
    'pc2': {
        var: data['pc'][var][1]
        for var in data['pc'].keys()
    },
}

ltime, llat, llon = data['pc1']['t'].shape

## 2. Load EOF data
eof_data = jl.load(path + 'EOF.joblib')

lev : np.ndarray = eof_data['lev']
eof1: np.ndarray = eof_data['EOF'][:, 0]
eof2: np.ndarray = eof_data['EOF'][:, 1]

# %% Section 3: Processing data
## 1. Formatting data
fmt = dp.Format(dims['lat'])

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: fmt.sym(data[pc][var])
        for var in data[pc].keys()
    }
    for pc in data.keys()
}

## 2. Chunking data
hann: np.ndarray = np.hanning(120)[:, None]

lsec: int = 120

chunking: dict[str, dict[str, np.ndarray]] = {
    pc: {
        var: np.array([
            sym[pc][var][i*60:i*60+120] * hann
            for i in range(5)
        ])
        for var in sym[pc].keys()
    }
    for pc in sym.keys()
}

# %% Compute the growth rate
## 1. Compute Growth Rate
sigma: dict[str, dict[str, np.ndarray]] = {
    'pc1': Growth_Rate(chunking['pc1']['cu'], chunking['pc1']['t']),
    'pc2': Growth_Rate(chunking['pc2']['cu'], chunking['pc2']['t']),
}

## 2. Compute Phase
phase: dict[str, dict[str, np.ndarray]] = {
    'pc1': Phase(chunking['pc1']['t'], chunking['pc1']['cu']),
    'pc2': Phase(chunking['pc2']['t'], chunking['pc2']['cu']),
}

## 3. Compute Coherence Square
coh2: dict[str, dict[str, np.ndarray]] = {
    'pc1': Coherence(chunking['pc1']['cu'], chunking['pc1']['t']),
    'pc2': Coherence(chunking['pc2']['cu'], chunking['pc2']['t']),
}

## 4. Fisher's Z transform
def z_transform(coh2):
    r = np.sqrt(coh2)

    return np.log((1+r)/(1-r))

Z_trans = {
    'pc1': z_transform(coh2['pc1']),
    'pc2': z_transform(coh2['pc2']),
}

# %% Section 4: Statistical test
criteria = {
    'pc1': Z_trans['pc1'].mean() + 1.64*Z_trans['pc1'].std(),
    'pc2': Z_trans['pc2'].mean() + 1.64*Z_trans['pc2'].std(),
}

sigma_filted = {
    'pc1': np.where(Z_trans['pc1'] >= criteria['pc1'], sigma['pc1'], np.nan),
    'pc2': np.where(Z_trans['pc2'] >= criteria['pc2'], sigma['pc2'], np.nan),
}

# %% Section 5: Plotting
## Design the plot
wn = np.fft.fftfreq(llon, d=1/llon)
fr = np.fft.fftfreq(120 , d=1/4)

wn_plot = np.fft.fftshift(wn)
fr_plot = np.fft.fftshift(fr)


plt.rcParams.update({
    'font.size': 12,
    'figure.titlesize': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'font.family': 'serif',
})

fr_ana, wn_ana = th.genDispersionCurves(Ahe=[90, 25, 8])

e_cond = np.where(wn_ana[3, 0, :] < 0)[0]

## Plotting
plt.figure(figsize=(8, 8))
c1 = plt.contourf(
    wn_plot, fr_plot[fr_plot > 0], 
    np.fft.fftshift(sigma_filted['pc1'])[fr_plot > 0],
    levels=np.linspace(-8, 8),
    cmap="BrBG_r",
    extend='both'
)
for i in range(3):
    plt.plot(wn_ana[3, i, e_cond], fr_ana[3, i, e_cond], 'k--')
    plt.plot(wn_ana[4, i, :], fr_ana[4, i, :], 'k--')
    plt.plot(wn_ana[5, i, :], fr_ana[5, i, :], 'k--')
plt.xticks(np.linspace(-14, 14, 8))
plt.yticks(np.linspace(0, 0.5, 6))
plt.xlim(-15, 15)
plt.ylim(0, 0.5)
plt.xlabel('Zonal Wavenumber')
plt.ylabel('Frequency [CPD]')
plt.title(f'{exp} PC1 EAPE Growth Rate')
cbar1 = plt.colorbar(
    c1,
    aspect=30,
    orientation='horizontal',
    label=r'$\sigma$ [day$^{-1}$]'
)
cbar1.set_ticks(np.linspace(-8, 8, 9))
plt.savefig(f'/home/b11209013/2024_Research/MPAS/GrowthRate/image/Cu/{exp}_PC1.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 8))
c2 = plt.contourf(
    wn_plot, fr_plot[fr_plot > 0],
    np.fft.fftshift(sigma_filted['pc2'])[fr_plot > 0],
    levels=np.linspace(-2.5, 2.5),
    cmap="BrBG_r",
    extend='both'
)
for i in range(3):
    plt.plot(wn_ana[3, i, e_cond], fr_ana[3, i, e_cond], 'k--')
    plt.plot(wn_ana[4, i, :], fr_ana[4, i, :], 'k--')
    plt.plot(wn_ana[5, i, :], fr_ana[5, i, :], 'k--')
plt.xticks(np.linspace(-14, 14, 8))
plt.yticks(np.linspace(0, 0.5, 6))
plt.xlim(-15, 15)
plt.ylim(0, 0.5)
plt.xlabel('Zonal Wavenumber')
plt.ylabel('Frequency [CPD]')
plt.title(f'{exp} PC2 EAPE Growth Rate')
cbar2 = plt.colorbar(
    c2,
    aspect=30,
    orientation='horizontal',
    label=r'$\sigma$ [day$^{-1}$]'
)
cbar2.set_ticks(np.linspace(-2.5, 2.5, 6))
plt.savefig(f'/home/b11209013/2024_Research/MPAS/GrowthRate/image/Cu/{exp}_PC2.png', dpi=300)
# %%
