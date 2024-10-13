# This code os for computing Q1 in MPAS experiments over tropical region

# import package
import numpy as np
import netCDF4 as nc
import pandas as pd

from multiprocessing import Pool


# ===================== #
# functions
# load data
def LoadData(case: str, var: str) -> np.ndarray:
    with nc.Dataset(
            f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/{var}_tropic.nc", "r"
            ) as f:
        data: np.ndarray = f.variables[f"{var}"][:]

    return data


# temp theta conversion
def Theta2Temp(theta: np.ndarray, plev: np.ndarray) -> np.ndarray:
    temp: np.ndarray = theta * (1000 / plev[None, :, None, None]) ** (-287.5 / 1004.5)
    return temp


# ===================== #
def main():
    """
    This code is for computing Q1
    """
    # ================= #
    # load data
    case: str = "NSC"
    path: str = f"/work/b11209013/MPAS/{case}/merged_data/tropic_data/"

    VarList: tuple[str] = (
            "w",
            "zgrid",
            "theta",
            "uReconstructMeridional",
            "uReconstructZonal",
            )

    # load coordinate
    with nc.Dataset(path + "zgrid_tropic.nc", "r") as f:
        lat: np.ndarray = f.variables["lat"][:]
        lon: np.ndarray = f.variables["lon"][:]
        lev: np.ndarray = f.variables["lev"][:]

    lonm, latm = np.meshgrid(lon, lat)

    data: dict[str, np.ndarray] = {}

    # load variables
    for var in VarList:
        data[var] = LoadData(case, var)

    for key in data.keys():
        print(data[key].shape)

    # ================= #
    # conversion between temperature and theta
    data["temp"] = Theta2Temp(data["theta"], lev)
    data.pop("theta")

    # ================= #
    # select date
    date = pd.date_range(start="2000-05-01 00:00", periods=492, freq="6h")
    InitIdx = np.where((date.year == 2000) & (date.month == 5) & (date.day == 31))[0][0]
    TermIdx = np.where((date.year == 2000) & (date.month == 8) & (date.day == 29))[0][-1]

    print(date[InitIdx])
    print(date[TermIdx])
    period = slice(InitIdx, TermIdx+1)

    origin_date = pd.Timestamp("1900-01-01 00:00:00")
    numeric_date = (date[period] - origin_date) / pd.Timedelta(hours=1)

    # align the data into the same length of time
    data_homo: dict[str, np.ndarray] = {
            "w": data["w"][period],
            "temp": data["temp"][period],
            "zgrid": data["zgrid"][4:],
            "uReconstructZonal": data["uReconstructZonal"][period],
            "uReconstructMeridional": data["uReconstructMeridional"][period],
            }

    ltime, llev, llat, llon = data_homo["w"].shape

    del data

    for key in data_homo.keys():
        print(data_homo[key].shape)

    print("Finish preparing data")

    # ================= #
    # compute Q1
    # # compute dry static energy
    dse: np.ndarray = 1004.5 * data_homo["temp"] + 9.8 * data_homo["zgrid"]

    # derivative on the dse
    # # derivative along time
    dse_dt = np.empty_like(dse)
    dt = 6 * 3600

    dse_dt[0] = (dse[1] - dse[0]) / (dt)
    dse_dt[1:-1] = (dse[2:] - dse[:-2]) / (2 * dt)
    dse_dt[-1] = (dse[-1] - dse[-2]) / (dt)

    # # derivative along longitude
    dse_dx = np.empty_like(dse)

    for i, l in enumerate(lat):
        dx = 2 * np.pi * 6.371e6 * np.cos(np.deg2rad(l)) / llon
        dse_dx[:, :, i, 0] = (dse[:, :, i, 1] - dse[:, :, i, 0]) / (dx)
        dse_dx[:, :, i, 1:-1] = (dse[:, :, i, 2:] - dse[:, :, i, :-2]) / (2 * dx)
        dse_dx[:, :, i, -1] = (dse[:, :, i, -1] - dse[:, :, i, -2]) / (dx)

    # # derivative along latitude
    dse_dy = np.empty_like(dse)
    dy = 2 * np.pi * 6.371e6 / 720

    dse_dy[:, :, 0] = (dse[:, :, 1] - dse[:, :, 0]) / (dy)
    dse_dy[:, :, 1:-1] = (dse[:, :, 2:] - dse[:, :, :-2]) / (2 * dy)
    dse_dy[:, :, -1] = (dse[:, :, -1] - dse[:, :, -2]) / (dy)

    # # derivative along level
    dse_dz = np.empty_like(dse)

    dse_dz[:, 0] = (dse[:, 1] - dse[:, 0]) / (
            data_homo["zgrid"][:, 1] - data_homo["zgrid"][:, 0]
            )
    dse_dz[:, 1:-1] = (dse[:, 2:] - dse[:, :-2]) / (
            data_homo["zgrid"][:, 2:] - data_homo["zgrid"][:, :-2]
            )
    dse_dz[:, -1] = (dse[:, -1] - dse[:, -2]) / (
            data_homo["zgrid"][:, -1] - data_homo["zgrid"][:, -2]
            )

    q1 = dse_dt\
            + (data_homo["uReconstructZonal"] * dse_dx)\
            + (data_homo["uReconstructMeridional"] * dse_dy)\
            + (data_homo["w"] * dse_dz)

    print("Finish computation of Q1")

    # ================= #
    # save nc file
    with nc.Dataset(path + "Q1/Q1.nc", "w") as f:
        lon_dim = f.createDimension("lon", llon)
        lat_dim = f.createDimension("lat", llat)
        lev_dim = f.createDimension("lev", llev)
        time_dim = f.createDimension("time", len(numeric_date))

        lon_var = f.createVariable("lon", np.float64, ("lon",))
        lon_var.units = "degrees_east, starts from -179.75"
        lon_var.long_name = "longitude"
        lon_var[:] = lon

        lat_var = f.createVariable("lat", np.float64, ("lat",))
        lat_var.units = "degrees_north, starts from south hemisphere"
        lat_var.long_name = "latitude"
        lat_var[:] = lat

        lev_var = f.createVariable("lev", np.float64, ("lev",))
        lev_var.units = "hPa"
        lev_var.long_name = "Given level for intepolation"
        lev_var[:] = lev

        time_var = f.createVariable("time", np.float64, ("time", ))
        time_var.units = "hours after 1900-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var[:] = numeric_date

        q1_var = f.createVariable("Q1", np.float64, ("time", "lev", "lat", "lon"))
        q1_var.units = "J/kg/s"
        q1_var.long_name = "Apparent Heat Source Q1"
        q1_var[:] = q1

    with nc.Dataset(path + "t/t.nc", "w") as f:
        lon_dim = f.createDimension("lon", llon)
        lat_dim = f.createDimension("lat", llat)
        lev_dim = f.createDimension("lev", llev)
        time_dim = f.createDimension("time", len(numeric_date))

        lon_var = f.createVariable("lon", np.float64, ("lon",))
        lon_var.units = "degrees_east, starts from -179.75"
        lon_var.long_name = "longitude"
        lon_var[:] = lon

        lat_var = f.createVariable("lat", np.float64, ("lat",))
        lat_var.units = "degrees_north, starts from south hemisphere"
        lat_var.long_name = "latitude"
        lat_var[:] = lat

        lev_var = f.createVariable("lev", np.float64, ("lev",))
        lev_var.units = "hPa"
        lev_var.long_name = "Given level for intepolation"
        lev_var[:] = lev

        time_var = f.createVariable("time", np.float64, ("time", ))
        time_var.units = "hours after 1900-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var[:] = numeric_date

        t_var = f.createVariable("t", np.float64, ("time", "lev", "lat", "lon"))
        t_var.units = "K"
        t_var.long_name = "Temperature"
        t_var[:] = data_homo["temp"]


# ===================== #
if __name__ == "__main__":
    main()
