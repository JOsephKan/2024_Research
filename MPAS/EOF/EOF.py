#%%
import numpy as np
import netCDF4 as nc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt


# %% load dataset
with nc.Dataset("/work/b11209013/2024_Research/MPAS/merged_data/CNTL/q1.nc") as f:
    dims = dict(
        lon = f.variables['lon'][:],
        lat = f.variables['lat'][:],
        lev = f.variables['lev'][:],
        time = f.variables['time'][:]
    )
    
    lat_lim = np.where((dims['lat'] >= -5) & (dims['lat'] <= 5))[0]
    
    dims['lat'] = dims['lat'][lat_lim]
    q1 = f.variables['q1'][:, :, lat_lim, :]

ltime, llev, llat, llon = q1.shape


# %% permute and reshape data
q1_rs = q1.transpose(1, 0, 2, 3).reshape(llev, -1)

scaler = StandardScaler()

data_scaled = ((q1_rs - q1_rs.mean())).T

n_eofs = 10

pca = PCA(n_components = n_eofs)

pca.fit(data_scaled)

eof_modes = pca.components_
pc_modes = pca.transform(data_scaled)
exp_var = pca.explained_variance_ratio_

# %% save data
with nc.Dataset("/work/b11209013/2024_Research/MPAS/PC/EOF.nc", "w") as f:
    f.createDimension("mode", n_eofs)
    f.createDimension("lev", llev)
    
    var_eof = f.createVariable("EOF", np.float64, ("mode", "lev"))
    var_eof.description="EOF modes of CNTL q1"
    var_eof[:] = eof_modes
    
    lev_var = f.createVariable("lev", np.float64, ("lev"))
    lev_var.description="Levels of EOF modes"
    lev_var[:] = dims['lev']
    
    mode_var = f.createVariable("mode", np.int32, ("mode"))
    mode_var.description="Mode number"
    mode_var[:] = np.linspace(1, n_eofs, n_eofs, dtype=int)
# %%
