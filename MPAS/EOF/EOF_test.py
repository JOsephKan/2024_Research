#%%
import numpy as np
import joblib as jl
from matplotlib import pyplot as plt

eof_data = jl.load('/work/b11209013/2024_Research/MPAS/PC/EOF.joblib')
lev = eof_data['lev']
eof = eof_data['EOF']

plt.plot(eof[:, 0], lev)
plt.plot(eof[:, 1], lev)
plt.gca().invert_yaxis()
# %%
