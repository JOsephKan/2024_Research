# This program aims to figure out the LW, SW, CU EAPE growth rate
# import packages
## import official packages
import sys
import numpy as np
import joblib as jl

from scipy.signal import convolve2d
from matplotlib import pyplot as plt

## import self-made packages
sys.path.append("/home/b11209013/Package")
import Theory as th        # type: ignore
import DataProcess as dp   # type: ignore
import SignalProcess as sp # type: ignore

# ==================== Part 1: Functions ==================== #
# Covariance
ft = sp.Fourier() # Fourier Transform object

## compute covariance between two datasets
def Covariance(
    data1: np.ndarray,
    data2: np.ndarray
    ) -> np.ndarray:
    
    # Compute the cross-spectrum
    cs: np.ndarray = ft.CrossSpectrum(data1, data2)
    
    # Define the smoothing kernel
    kernel: np.ndarray = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # 2D kernel
    
    # Smooth using 2D convolution
    cs_smooth = convolve2d(
        cs.real, kernel, mode='same') + 1j * convolve2d(cs.imag, kernel, mode='same'
)
    
    return cs_smooth

## Phase
def Phase(
    data1: np.ndarray,
    data2: np.ndarray
    )-> np.ndarray:
    
    cs: np.ndarray = np.array([
        Covariance(d1, d2)
        for d1, d2 in zip(data1, data2)
    ])
    
    phase: np.ndarray = np.atan2(cs.imag, cs.real)
    
    return phase

## Growth Rate
def Growth_Rate(
    data1: np.ndarray,
    data2: np.ndarray
    )-> np.ndarray:
    
    var: np.ndarray = np.array([
        Covariance(data2[i], data2[i]).real
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    cov: np.ndarray = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    
    sigma: np.ndarray = 2*np.real(cov) / var
    
    return sigma

## Coherence Square
def Coherence(
    data1: np.ndarray,
    data2: np.ndarray
    )-> np.ndarray:
    
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

def main():
# ==================== Part 2: Load Data ==================== #
    # case name
    case: str = 'CNTL'

    # PC file name
    pc_fname : str = f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.joblib'

    # EOF file name
    eof_fname: str = f'/work/b11209013/2024_Research/MPAS/PC/CNTL_EOF.joblib'
    
    # linear response function file name
    lrf_fname: str = f'/work/b11209013/2024_Research/MPAS/LRF/lrf_{case}.joblib'

    # load PC file
    data: dict[str] = jl.load(pc_fname)

    lon  : np.ndarray = data['lon']
    lat  : np.ndarray = data['lat']
    time : np.ndarray = data['time']
    pc1  : dict[str] = data['pc1'] # dictionary for t, q1, qv pc1
    pc2  : dict[str] = data['pc2'] # dictionary for t, q1, qv pc2

    tpc  : np.ndarray = np.stack([pc1['t'], pc2['t']])
    qvpc : np.ndarray = np.stack([pc1['qv'], pc2['qv']])

    ltime, llat, llon = pc1['t'].shape

    # load LRF files
    eof : np.ndarray = jl.load(eof_fname)
    eof_f2 = eof[:, :2]

    # load LRF file
    lrf: dict[str] = jl.load(lrf_fname)

    lev: np.ndarray = lrf['lev']
    lw : np.ndarray = np.where(np.isnan(lrf['lw_lrf'])==True, 0, lrf['lw_lrf'])
    sw : np.ndarray = np.where(np.isnan(lrf['sw_lrf'])==True, 0, lrf['sw_lrf'])
    cu : np.ndarray = np.where(np.isnan(lrf['cu_lrf'])==True, 0, lrf['cu_lrf'])

    llev = len(lev)
    
# ==================== Part 3: Construct state vector ==================== #
    # reshape PC
    tpc_rs : np.ndarray = tpc.reshape( (2, ltime*llat*llon))
    qvpc_rs: np.ndarray = qvpc.reshape((2, ltime*llat*llon))
    
    # form PC with EOF1 and EOF2
    t_state : np.ndarray = eof_f2 @ tpc_rs
    qv_state: np.ndarray = eof_f2 @ qvpc_rs

    # form total state vector
    state_vec: np.ndarray = np.concatenate((t_state, qv_state), axis=0)

# ==================== Part 4: Compute corresponding heating ==================== #
    # compute heating
    lw_heating: np.ndarray = (lw @ state_vec)
    sw_heating: np.ndarray = (sw @ state_vec)
    cu_heating: np.ndarray = (cu @ state_vec)[:18]
    
    # separate into different vertical structures
    eof1: np.ndarray = np.matrix(eof_f2[:, 0]).T # EOF1
    eof2: np.ndarray = np.matrix(eof_f2[:, 1]).T # EOF2

    heating: dict[str, np.ndarray] = {
        'pc1':{
            'lw': np.array(np.linalg.inv(eof1.T @ eof1) @ (eof1.T @ lw_heating)).squeeze(),
            'sw': np.array(np.linalg.inv(eof1.T @ eof1) @ (eof1.T @ sw_heating)).squeeze(),
            'cu': np.array(np.linalg.inv(eof1.T @ eof1) @ (eof1.T @ cu_heating)).squeeze()
        },
        'pc2':{
            'lw': np.array(np.linalg.inv(eof2.T @ eof2) @ (eof2.T @ lw_heating)).squeeze(),
            'sw': np.array(np.linalg.inv(eof2.T @ eof2) @ (eof2.T @ sw_heating)).squeeze(),
            'cu': np.array(np.linalg.inv(eof2.T @ eof2) @ (eof2.T @ cu_heating)).squeeze(),
        }
    }

    # reshape back to original shape
    heating_rs: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: heating[pc][var].reshape((ltime, llat, llon))
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }

# ==================== Part 5: Processing data ==================== #
    # Format data
    fmt = dp.Format(lat)
    
    sym: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: fmt.sym(heating_rs[pc][var])
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }

    sym['pc1']['t'] = fmt.sym(pc1['t'])
    sym['pc2']['t'] = fmt.sym(pc2['t'])

    # Chunking data
    Hann = np.hanning(120,)[:, None]

    window: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: np.array([
                sym[pc][var][i*60:i*60+120] * Hann
                for i in range(5)
            ])
            for var in ['t', 'lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
# ==================== Part 6: Compute Variables ==================== #
    # Phase
    phase: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: Phase(window[pc]['t'], window[pc][var])
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
    
    # Growth Rate
    sigma: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: Growth_Rate(window[pc][var], window[pc]['t'])
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }

    # Coherence Square
    Coh2: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: Coherence(window[pc][var], window[pc]['t'])
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
    
# ==================== Part 7: Statistical test ==================== #
    # Fisher's transform
    def z_transform(coh2):
        r = np.sqrt(coh2)

        return np.log((1+r)/(1-r))

    z_trans: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: z_transform(Coh2[pc][var])
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
    
    # criteria for statistical test
    crit: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: z_trans[pc][var].mean() + 1.96*z_trans[pc][var].std()
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
    
    # Filter out the significant growth rate
    sigma_filted: dict[str, dict[str, np.ndarray]] = {
        pc: {
            var: np.where(z_trans[pc][var] >= crit[pc][var], sigma[pc][var], np.nan)
            for var in ['lw', 'sw', 'cu']
        }
        for pc in ['pc1', 'pc2']
    }
    
# ==================== Part 8: Plotting ==================== #

    # Plot setting
    plt.rcParams.update({
        'font.size': 12,
        'figure.titlesize': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'font.family': 'serif',
    })
    
    wn = np.linspace(-359, 360, 720).astype(int)
    fr = np.linspace(1/30, 2, 60)
    
    # thoretical dispersion relation
    fr_ana, wn_ana = th.genDispersionCurves(Ahe=[90, 25, 8])

    r_cond = np.where(wn_ana[3, 0] < 0)[0] # condition that satisfied with equatorial Rossby waves
    
    # Plotting
    
    for i, var in enumerate(['lw', 'sw', 'cu']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (17, 8))
        c1 = ax1.contourf(
            wn, fr,
            sigma_filted['pc1'][var],
            levels = np.linspace(-6, 6),
            cmap='BrBG_r',
            extend='both'
        )
        for i in range(3):
            ax1.plot(wn_ana[3, i, r_cond], fr_ana[3, i, r_cond], 'k--')
            ax1.plot(wn_ana[4, i, :], fr_ana[4, i, :], 'k--')
            ax1.plot(wn_ana[5, i, :], fr_ana[5, i, :], 'k--')

        ax1.set_xticks(np.linspace(-14, 14, 8))
        ax1.set_yticks(np.linspace(0, 0.5, 6))
        ax1.set_xlim(-15, 15)
        ax1.set_ylim(0, 0.5)
        ax1.set_xlabel('Zonal Wavenumber')
        ax1.set_ylabel('Frequency [CPD]')
        ax1.set_title(f'{case} PC1 {var} EAPE Growth Rate')
        cbar1 = plt.colorbar(
            c1,
            ax=ax1,
            aspect=30,
            orientation='horizontal',
            label=r'$\sigma$ [day$^{-1}$]'
        )
        cbar1.set_ticks(np.linspace(-6, 6, 7))
        
        c2 = ax2.contourf(
            wn, fr, 
            sigma_filted['pc2'][var],
            levels=np.linspace(-1.8, 1.8),
            cmap="BrBG_r",
            extend='both'
        )
        for i in range(3):
            ax2.plot(wn_ana[3, i, r_cond], fr_ana[3, i, r_cond], 'k--')
            ax2.plot(wn_ana[4, i, :], fr_ana[4, i, :], 'k--')
            ax2.plot(wn_ana[5, i, :], fr_ana[5, i, :], 'k--')
        ax2.set_xticks(np.linspace(-14, 14, 8))
        ax2.set_yticks(np.linspace(0, 0.5, 6))
        ax2.set_xlim(-15, 15)
        ax2.set_ylim(0, 0.5)
        ax2.set_xlabel('Zonal Wavenumber')
        ax2.set_ylabel('Frequency [CPD]')
        ax2.set_title(f'{case} PC2 {var} EAPE Growth Rate')
        cbar2 = plt.colorbar(
            c2,
            ax=ax2,
            aspect=30,
            orientation='horizontal',
            label=r'$\sigma$ [day$^{-1}$]'
        )
        cbar2.set_ticks(np.linspace(-1.8, 1.8, 7))
        
        plt.savefig(f'/home/b11209013/2024_Research/MPAS/Separated_GrowthRate/{case}_{var}_growthrate.png', dpi=300)
        plt.close()

# ===================== Execution ======================== #
if __name__ == '__main__':
    main()
