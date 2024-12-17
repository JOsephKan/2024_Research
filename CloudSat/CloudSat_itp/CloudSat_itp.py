# This program is to interpolate the CloudSat data to ERA5 grid
# environment setting
# # import package
import os;
import sys;
import numpy as np;
import joblib as jl;
import netCDF4 as nc;

# # accept environmental arguments
year: int = int(sys.argv[1]);
date: int = int(sys.argv[2]);

def check_file_exists(filepath):
    if not os.path.exists(filepath):
#         print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code


# load data
path: str = "/work/b11209013/2024_Research/CloudSat/";

check_file_exists(path + f"CloudSat_chunked/{year}_{date}.joblib")

# # load CloudSat data
cs_data: dict[str, np.ndarray] = jl.load(path + f"CloudSat_chunked/{year}_{date}.joblib");

# # load ERA5 geopotential data
era5_data: dict[str, np.ndarray] = jl.load(path + f"z_itp/{year}.joblib");


