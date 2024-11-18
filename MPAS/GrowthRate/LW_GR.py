# This program is to compute the growth rate of the LW radiation
# import package
import numpy as np
import netCDF4 as nc
import joblib as jl

from matplotlib import pyplot as plt

# ========================== #
# Functions
# # Covariance

# ========================== #
#
def main():
    # load data
    fpath: str='/work/b11209013/2024_Research/MPAS/'

    ## EOF structure
    eof: np.ndarray = jl.load(f'{fpath}PC/CNTL_EOF.joblib')

    ## load longwave heating
    pc: dict[str] = jl.load(f'{fpath}PC/CNTL_PC.joblib')

    tpc1: np.ndarray = pc['pc1']['t']
    tpc2: np.ndarray = pc['pc2']['t']

    

# =========================== #
if __name__ == "__main__":
    main()
