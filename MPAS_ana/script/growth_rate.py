# This program is to compute growth rate
# import package
import sys
import numpy as np
import netCDF4 as nc

sys.path.append("/home/b11209013/Package/")
import Theory as th
import DataProcess as dp
import SignalProcess as sp

# ====================== #
# function
# # load data
def load_data(case: str, var: str) -> np.ndarray:
    with nc.Dataset(f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/{var}/{var}PC.nc", "r") as f:
        data = f.variables[var][:] 

    return data

# ====================== #
def main():
    # Load data
    path = "/work/b11209013/MPAS/"
    var_list = ["Q1", "t"]
    case_list = ["CNTL", "NCRF", "NSC", "SD15"]

    # # load coordinate information
    with nc.Dataset(path+"CNTL/merged_data/tropic_data/Q1/Q1PC.nc", "r") as f:
        lat = f.variables["lat"][:]
        lon = f.variables["lon"][:]

    # # load PC information
    data = {}
    for case in case_list:
        data[case] = {}

        for var in var_list:
            data[case][var] = load_data(case, var)

    # ================== #
    # process data
    # # format data
    sym = {}
    asy = {}

    fmt = dp.Format(lat)

    for case in case_list:
        sym[case] = {}; asy[case] = {}

        for var in var_list:
            sym[case][var] = fmt.sym(data[case][var])
            asy[case][var] = fmt.asy(data[case][var])

    print(sym["CNTL"]["Q1"].shape)

# ====================== #
if __name__ == "__main__":
    main()
