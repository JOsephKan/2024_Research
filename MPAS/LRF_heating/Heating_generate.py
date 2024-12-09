# This program is to generate heaing profile from given temperature and moisture rprofile
# %% impport package
## 1. import package
import numpy as np
import joblib as jl

## 2. experiment name
exp: str = 'NSC'

# %% section 2: 
# Load data
## 1. Load PC data
data_pc: dict[str, dict[str, np.ndarray]] = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/{exp}_PC.joblib')

pc: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var : data_pc['pc'][var][0]
        for var in data_pc['pc'].keys()
    },
    'pc2': {
        var : data_pc['pc'][var][1]
        for var in data_pc['pc'].keys()
    }
}

ltime, llat, llon = pc['pc1']['t'].shape

## 2. Load EOF profile
data_eof: dict[str, dict[str, np.ndarray]] = jl.load(f'/work/b11209013/2024_Research/MPAS/PC/EOF.joblib')

lev : np.ndarray = data_eof['lev']
eof1: np.ndarray = data_eof['EOF'][:, 0]
eof2: np.ndarray = data_eof['EOF'][:, 1]

## 3. load LRF file
data_lrf: dict[str, np.ndarray] = jl.load(f'/home/b11209013/2024_Research/MPAS/LRF_construct/LRF_file/LRF_{exp}.joblib')

lw : np.ndarray = np.where(np.isnan(data_lrf['lw']) , 0, data_lrf['lw'])
sw : np.ndarray = np.where(np.isnan(data_lrf['sw']) , 0, data_lrf['sw'])
cu : np.ndarray = np.where(np.isnan(data_lrf['cu']) , 0, data_lrf['cu'])
tot: np.ndarray = np.where(np.isnan(data_lrf['tot']), 0, data_lrf['tot'])

# %% section 3: generate state vector
## 1. Constructing vertical profile of temperature and moisture
data_flat: dict[str, dict[str, np.ndarray]] = {
    pc_: {
        var: pc[pc_][var].flatten()
        for var in pc[pc_].keys()
    }
    for pc_ in pc.keys()
}

vert_prof: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        't' : eof1[:, None] @ data_flat['pc1']['t'][None, :],
        'qv': eof1[:, None] @ data_flat['pc1']['qv'][None, :]
    },
    'pc2': {
        't' : eof2[:, None] @ data_flat['pc2']['t'][None, :],
        'qv': eof2[:, None] @ data_flat['pc2']['qv'][None, :]
    }
}

## 2. constructing state vector
state_vector: np.ndarray = np.concatenate(
    [vert_prof['pc1']['t']+vert_prof['pc2']['t'], vert_prof['pc1']['qv']+vert_prof['pc2']['qv']],
    axis = 0
)

# %% section 4: generate heating PC
## 1. generate heating profile
heating: dict[str, np.ndarray] = {
    'lw' : lw  @ state_vector,
    'sw' : sw  @ state_vector,
    'cu' : cu  @ state_vector,
    'tot': tot @ state_vector
}

## 2. generate heating PC
heating_pc: dict[str, dict[str, np.ndarray]] = {
    'pc1': {
        var: np.linalg.inv(eof1[None, :] @ eof1[:, None]) @ (eof1[None, :] @ heating[var])
        for var in heating.keys()
    },
    'pc2': {
        var: np.linalg.inv(eof2[None, :] @ eof2[:, None]) @ (eof2[None, :] @ heating[var])
        for var in heating.keys()
    }
}

# %% section 5: save heating profile
jl.dump(heating_pc, f'/home/b11209013/2024_Research/MPAS/LRF_heating/heating_file/{exp}_heating.joblib')
