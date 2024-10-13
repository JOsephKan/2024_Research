# This program aims to acquire vertical EOF structure and principal structures of both Q1 and temperature
# import package
import numpy as np
import netCDF4 as nc

from multiprocessing import Pool
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# ====================== #
# functions
def LoadData(case: str, var: str) -> np.ndarray:

    fname: str = f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/{var}/{var}.nc"

    data = {}
    with nc.Dataset(fname, "r") as f:
        data["lat"] = f.variables["lat"][:]
        data["lon"] = f.variables["lon"][:]
        data["lev"] = f.variables["lev"][:]
        data[var] = f.variables[var][:]

    return data

# normal equation
def NormalEquation(arr: np.ndarray, eof: np.ndarray) -> np.ndarray:
    xTx = np.linalg.inv(np.matmul(eof.T, eof))
    op = np.matmul(xTx, eof.T)
    normal = np.matmul(op, arr)

    return normal

# function for computing EOF
def EmpOrthFunc(arr: np.ndarray) -> np.ndarray:
    CovMat = np.matmul(arr, arr.T)/arr.shape[1] # covariance matrix

    eigvals, eigvecs = np.linalg.eig(CovMat) # compute eigenvalues and eigenvectors

    ExpVar = eigvals / eigvals.sum() # compute explained variance
    EOF = -(eigvecs - eigvecs.mean())/eigvecs.std() # normalized eigenvectors into EOF
    PC = NormalEquation(arr, EOF)

    return ExpVar, EOF, PC

# ====================== #

def main():
    # ====================== #
    # load data
    case_list = ["CNTL", "NCRF", "NSC", "SD15"]
    VarList = ["Q1", "t"]
    data = {}

    # # load data
    for var in VarList:
        data[var] = {}

        for case in case_list:
            data[var][case] = LoadData(case, var)

    print("Finish Loading")

    # ====================== #
    # transpose and reshape data
    data_reshape = {}

    for var in VarList:
        data_reshape[var] = {}

        for case in case_list:
            trans = data[var][case][var].transpose((1, 0, 2, 3))

            llev, ltime, llat, llon = trans.shape
            data_reshape[var][case] = trans.reshape((llev, ltime*llat*llon))

    print("Finish reshape")

    # ======================= #
    # EOF
    ExpVar = {}; EOF = {}; Q1PC = {}; TPC = {}

    for case in case_list:
        ExpVar[case], EOF[case], Q1PC[case] = EmpOrthFunc(np.array(data_reshape["Q1"][case]))

    for case in case_list:
        TPC[case] = NormalEquation(np.array(data_reshape["t"][case]), np.array(EOF[case]))
    # get reshape PCs
    PCs = {
            "tpc1": {}, "tpc2": {},
            "q1pc1": {}, "q1pc2": {},
            }

    for case in case_list:
        ltime, llev, llat, llon = data["Q1"][case]["Q1"].shape

        PCs["tpc1"][case] = TPC[case][0, :].reshape((ltime, llat, llon))
        PCs["tpc2"][case] = TPC[case][1, :].reshape((ltime, llat, llon))
        PCs["q1pc1"][case] = Q1PC[case][0, :].reshape((ltime, llat, llon))
        PCs["q1pc2"][case] = Q1PC[case][1, :].reshape((ltime, llat, llon))

    print("Finish PC compute")

    # ======================== #
    # save as PC
    for case in case_list:
        Fpath = f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/"
        with nc.Dataset(Fpath+"Q1/Q1PC.nc", "w") as f:
            lat_dim = f.createDimension("lat", llat)
            lon_dim = f.createDimension("lon", llon)
            time_dim = f.createDimension("time", None)

            lat_var = f.createVariable("lat", np.float64, ("lat", ))
            lat_var.standard_name = "lat"
            lat_var.long_name = "latitude"
            lat_var.units = "degrees_north"
            lat_var.axis="Y"

            lon_var = f.createVariable("lon", np.float64, ("lon", ))
            lon_var.standard_name = "lon"
            lon_var.long_name = "longitude"
            lon_var.units = "degrees_east"
            lon_var.axis = "X"

            q1pc1_var = f.createVariable("q1pc1", np.float64, ("time", "lat", "lon"))
            q1pc1_var.long_name = "Q1 PC1"
            q1pc1_var.units = "J/(kg*s)"

            q1pc2_var = f.createVariable("q1pc2", np.float64, ("time", "lat", "lon"))
            q1pc2_var.long_name = "Q1 PC2"
            q1pc2_var.units = "J/(kg*s)"


            lat_var[:] = data["Q1"][case]["lat"]
            lon_var[:] = data["Q1"][case]["lon"]
            q1pc1_var[:] = PCs["q1pc1"][case]
            q1pc2_var[:] = PCs["q1pc2"][case]

        with nc.Dataset(Fpath+"t/tPC.nc", "w") as f:
            lat_dim = f.createDimension("lat", llat)
            lon_dim = f.createDimension("lon", llon)
            time_dim = f.createDimension("time", None)

            lat_var = f.createVariable("lat", np.float64, ("lat", ))
            lat_var.standard_name = "lat"
            lat_var.long_name = "latitude"
            lat_var.units = "degrees_north"
            lat_var.axis="Y"

            lon_var = f.createVariable("lon", np.float64, ("lon", ))
            lon_var.standard_name = "lon"
            lon_var.long_name = "longitude"
            lon_var.units = "degrees_east"
            lon_var.axis = "X"

            tpc1_var = f.createVariable("tpc1", np.float64, ("time", "lat", "lon"))
            tpc1_var.long_name = "T PC1"
            tpc1_var.units = "K"

            tpc2_var = f.createVariable("tpc2", np.float64, ("time", "lat", "lon"))
            tpc2_var.long_name = "T PC2"
            tpc2_var.units = "K"

            lat_var[:] = data["t"][case]["lat"]
            lon_var[:] = data["t"][case]["lon"]
            q1pc1_var[:] = PCs["tpc1"][case]
            q1pc2_var[:] = PCs["tpc2"][case]

        print("Finish output "+case)

# ======at================ #
if __name__ == "__main__":
    main()
