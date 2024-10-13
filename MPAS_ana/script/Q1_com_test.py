# This code os for computing Q1 in MPAS experiments over tropical region

# import package
import numpy as np
import netCDF4 as nc

from pandas import date_range
from multiprocessing import Pool

# ===================== # 
# functions
# load data
def LoadData(var: str) -> np.ndarray:
    with nc.Dataset(
            f"/work/b11209013/ERA5_tropic/{var}_tropic.nc",
        "r") as f:
        data: np.ndarray = f.variables[f"{var}"][:]
    
    return data

# temp theta conversion
def Theta2Temp(theta: np.ndarray, plev: np.ndarray) -> np.ndarray:
    temp: np.ndarray = theta * (1000/plev[None, :, None, None])**(-287.5/1004.5)
    return temp

# ===================== #
def main():
    """
    This code is for computing Q1
    """
    # ================= #
    # load data
    path: str = "/work/b11209013/ERA5_tropic/"

    VarList: tuple[str] = ("z", "t", "u", "v", "w")

    # load coordinate
    with nc.Dataset(path+"z_tropic.nc", "r") as f:
        lat: np.ndarray = f.variables["lat"][:]
        lon: np.ndarray = f.variables["lon"][:]
        lev: np.ndarray = np.array([100000, 92500, 85000, 70000, 50000, 25000, 20000, 10000])
    
    lonm, latm = np.meshgrid(lon, lat)

    # load variables
    data = {}
    for var in VarList:
        data[var] = LoadData(var)

    for key in data.keys():
        print(data[key].shape)

    ltime, llev, llat, llon = data["w"].shape

    print("Finish preparing data")

    # ================= #
    # compute Q1
    # # compute dry static energy
    dse: np.ndarray = 1004.5*data["t"] + data["z"]

    # derivative on the dse
    # # derivative along time
    dse_dt = np.empty_like(dse)
    dt = 86400

    dse_dt[0] = (dse[1]-dse[0])/(dt)
    dse_dt[1:-1] = (dse[2:]-dse[:-2])/(2*dt)
    dse_dt[-1] = (dse[-1]-dse[-2])/(-dt)

    # # derivative along longitude
    dse_dx = np.empty_like(dse)

    for i, l in enumerate(lat):
        dx = 2*np.pi*6.371e6*np.cos(np.deg2rad(l))/llon
        dse_dx[:, :, i, 0] = (dse[:, :, i, 1]-dse[:, :, i, 0])/(dx)
        dse_dx[:, :, i, 1:-1] = (dse[:, :, i, 2:]-dse[:, :, i, :-2])/(2*dx)
        dse_dx[:, :, i, -1] = (dse[:, :, i, -1]-dse[:, :, i, -2])/(-dx)

    # # derivative along latitude
    dse_dy = np.empty_like(dse)
    dy = 2*np.pi*6.371e6/576

    dse_dy[:, :, 0] = (dse[:, :, 1]-dse[:, :, 0])/(dy)
    dse_dy[:, :, 1:-1] = (dse[:, :, 2:]-dse[:, :, :-2])/(2*dy)
    dse_dy[:, :, -1] = (dse[:, :, -1]-dse[:, :, -2])/(-dy)

    # # derivative along level
    dse_dz = np.empty_like(dse)
    
    dse_dz[:, 0] = (dse[:, 1]-dse[:, 0])/(lev[1]-lev[0])
    dse_dz[:, 1:-1] = (dse[:, 2:]-dse[:, :-2])/(lev[2:]-lev[:-2])[None, :, None, None]
    dse_dz[:, -1] = (dse[:, -1]-dse[:, -2])/(lev[-1]-lev[-2])

    q1 = dse_dt\
            + (data["u"] * dse_dx)\
            + (data["v"] * dse_dy)\
            + (data["w"] * dse_dz)

    print("Finish computation of Q1")

    # ================= #
    # save nc file
    with nc.Dataset(path+"Q1.nc", "w") as f:
        lon_dim = f.createDimension("lon", llon)
        lat_dim = f.createDimension("lat", llat)
        lev_dim = f.createDimension("lev", llev)
        time_dim = f.createDimension("time", ltime)

        lon_var = f.createVariable("lon", np.float64, ("lon", ))
        lon_var.units = "degrees_east, starts from -179.75" 
        lon_var.long_name = "longitude"
        lon_var[:] = lon

        lat_var = f.createVariable("lat", np.float64, ("lat", ))
        lat_var.units = "degrees_north, starts from south hemisphere"
        lat_var.long_name = "latitude"
        lat_var[:] = lat

        lev_var = f.createVariable("lev", np.float64, ("lev", ))
        lev_var.units = "hPa"
        lev_var.long_name = "Given level for intepolation"
        lev_var[:] = lev

        q1_var = f.createVariable("Q1", np.float64, ("time", "lev", "lat", "lon"))
        q1_var.units = "J/kg/s"
        q1_var.long_name = "Apparent Heat Source Q1"
        q1_var[:] = q1

# ===================== #
if __name__ == "__main__":
    main()
