# This program is to generate heating profile for difference vertical modes
# %% section 1: environment setting
## import package
import numpy as np
import joblib as jl
import matplotlib.pyplot as plt
from panel import state

## environment setting
exp: str = 'CNTL'

# %% section 2: load data
## load PC data
pc_path = f"/work/b11209013/2024_Research/MPAS/PC/{exp}_PC.joblib"

pc_data = jl.load(pc_path)

### Dimension information
dims: dict[str, np.ndarray] = dict(
    lon  = pc_data['lon'],
    lat  = pc_data['lat'],
    time = pc_data['time']
)

### PC data
data: dict[str, dict[str, np.ndarray]] = dict(
    pc1= {
        var: pc_data['pc'][var][0]
        for var in pc_data['pc']
    },
    pc2= {
        var: pc_data['pc'][var][1]
        for var in pc_data['pc']
    }
)

ltime, llat, llon = data['pc1']['t'].shape

## Load LRF function
lrf_path = f"/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/LRF_{exp}.joblib"

### LRF data
lrf_data = jl.load(lrf_path)

lrf_data = {
    var: np.where(np.isnan(lrf_data[var]), 0, lrf_data[var])
    for var in lrf_data
}

## Load EOF data
eof_path = f"/work/b11209013/2024_Research/MPAS/PC/EOF.joblib"

eof_data = jl.load(eof_path)

dims['lev'] = eof_data['lev']
eof_data = dict(
    eof1 = eof_data['EOF'][:, 0][:, None],
    eof2 = eof_data['EOF'][:, 1][:, None]
)

# %% Construct vertical profile
## permute and reshape
data['pc1'] = {
    var: data['pc1'][var].reshape(1, -1)
    for var in data['pc1']
}

data['pc2'] = {
    var: data['pc2'][var].reshape(1, -1)
    for var in data['pc2']
}

vert_prof = dict(
    pc1 = {
        var: eof_data['eof1'] @ data['pc1'][var]
        for var in data['pc1']
    },
    pc2 = {
        var: eof_data['eof2'] @ data['pc2'][var]
        for var in data['pc2']
    }
)
## construct state vector
state_vec = np.concatenate([
    vert_prof['pc1']['t'] + vert_prof['pc2']['t'],
    vert_prof['pc1']['qv'] + vert_prof['pc2']['qv'],
], axis=0)

print(vert_prof['pc2']['t'])

# %% Construct heating profile
## heating profile
heating = dict(
    lw = (lrf_data['lw'] @ state_vec) - (lrf_data['lw'] @ state_vec).mean(),
    sw = (lrf_data['sw'] @ state_vec) - (lrf_data['lw'] @ state_vec).mean(),
    cu = (lrf_data['cu'] @ state_vec) - (lrf_data['lw'] @ state_vec).mean(),
)

norm_equ = lambda x, eof: np.linalg.inv(eof.T @ eof) @ eof.T @ x

heating = dict(
    pc1 = dict(
        lw = (np.dot(eof_data['eof1'].T, heating['lw'])).squeeze(),
        sw = (np.dot(eof_data['eof1'].T, heating['sw'])).squeeze(),
        cu = (np.dot(eof_data['eof1'].T, heating['cu'])).squeeze()
        ),
    pc2 = dict(
        lw = (np.dot(eof_data['eof2'].T, heating['lw'])).squeeze(),
        sw = (np.dot(eof_data['eof2'].T, heating['sw'])).squeeze(),
        cu = (np.dot(eof_data['eof2'].T, heating['cu'])).squeeze()
        ),
)
print(heating['pc1']['lw'])
## reshape heating profile
heating = {
    mode: {
        var: heating[mode][var].reshape(ltime, llat, llon)
        for var in heating[mode]
    }
    for mode in heating
}

print(heating['pc1']['lw'].shape)
# %% examining heating
## plot heating profile

plt.contourf(dims['lon'], dims['lat'], heating['pc1']['lw'].mean(axis=0), cmap='BrBG_r')
plt.title('LRF LW PC1')
plt.colorbar()
plt.show()

plt.contourf(dims['lon'], dims['lat'], data['pc1']['lw'].reshape((ltime, llat, llon)).mean(axis=0), cmap='BrBG_r')
plt.title('MPAS LW PC1')
plt.colorbar()
plt.show()

# %%
