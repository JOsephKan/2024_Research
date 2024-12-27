# This program is to interpolate the CloudSat data to ERA5 grid
# environment setting
# # import package
import os;
import sys;
import numpy as np;
import joblib as jl;
import netCDF4 as nc;

from scipy.interpolate import interp1d;

# # accept environmental arguments
year: int = int(sys.argv[1]);
date: int = int(sys.argv[2]);

# functions
# check file path exist
def check_file_exists(filepath):
    if not os.path.exists(filepath):
#         print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code

# interpolate the CloudSat to ERA5 grid
def itp(
    cs_data: dict[str, np.ndarray],
    era5_data: dict[str, np.ndarray],
) -> np.ndarray:


    qlw = np.empty((cs_data["lat"].size, cs_data["lon"].size, era5_data["lev"].size));
    qsw = np.empty((cs_data["lat"].size, cs_data["lon"].size, era5_data["lev"].size));

    for la in range(cs_data["lat"].size):
        for lo in range(cs_data["lon"].size):
            if len(cs_data["hgt"][la, lo]) == 0:
                qlw[la, lo] = np.nan;
                qsw[la, lo] = np.nan;
                continue

            qlw_itp: list[np.ndarrry] = [];
            qsw_itp: list[np.ndarray] = [];

            for l in range(len(cs_data["hgt"][la, lo])):
                qlw_func_itp = interp1d(cs_data["hgt"][la, lo][l], cs_data["qlw"][la, lo][l], kind="linear");
                qsw_func_itp = interp1d(cs_data["hgt"][la, lo][l], cs_data["qsw"][la, lo][l], kind="linear");

                qlw_itp.append(qlw_func_itp(era5_data["z"][date-1, :, la, lo]/9.81));
                qsw_itp.append(qsw_func_itp(era5_data["z"][date-1, :, la, lo]/9.81));
            
            qlw_mean = np.nanmean(np.array(qlw_itp), axis=0);
            qsw_mean = np.nanmean(np.array(qsw_itp), axis=0);

            qlw[la, lo] = qlw_mean;
            qsw[la, lo] = qsw_mean;

    return qlw, qsw



# load data
path: str = "/work/b11209013/2024_Research/CloudSat/";

check_file_exists(path + f"CloudSat_chunked/{year}_{date}.joblib")

# # load CloudSat data
cs_data: dict[str, np.ndarray] = jl.load(path + f"CloudSat_chunked/{year}_{date}.joblib");

# # load ERA5 geopotential data
era5_data: dict[str, np.ndarray] = jl.load(path + f"z_itp/{year}.joblib");

output_dict: dict[str, np.ndarray] = {
    "lon": cs_data["lon"],
    "lat": cs_data["lat"],
    "lev": era5_data["lev"],
    "qlw": itp(cs_data, era5_data)[0],
    "qsw": itp(cs_data, era5_data)[1],
};


jl.dump(output_dict, f"/work/b11209013/2024_Research/CloudSat/CloudSat_itp/{year}_{date}.joblib", compress=('zlib', 3));
print(f"{year}_{date} CloudSat interpolated");
