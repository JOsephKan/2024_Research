# %% 
# import package
import sys
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('/home/b11209013/Package')
import DataProcess as dp
import Spectral_Analysis as sa
# %%
# load the data
cases: list[str] = ['CNTL', 'NCRF', 'NSC']

data: dict[str, dict[str, np.ndarray]] = {
    'pc1': {},
    'pc2': {},
}

for case in cases:
    with nc.Dataset(f'/work/b11209013/2024_Research/MPAS/PC/{case}_PC.nc') as f:
        lat = f['lat'][:]

        data['pc1'][case] = f['q1'][0].transpose(2, 1, 0)
        data['pc2'][case] = f['q1'][1].transpose(2, 1, 0)


ltime, llat, llon = data['pc1']['CNTL'].shape


fmt = dp.Format(lat)

sym: dict[str, dict[str, np.ndarray]] = {
    pc: {
        case: fmt.sym(data[pc][case])
        for case in cases
    }
    for pc in ['pc1', 'pc2']
}

# %%
q1_sa = sa.SpectralAnalysis(sym['pc1']['CNTL'])

plt.contourf(np.log(q1_sa.power_spectrum()))
plt.xlim(345, 375)
# %%
