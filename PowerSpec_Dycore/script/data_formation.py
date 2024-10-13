# This program is to form the data into pkl file
# Basic setting
# # import package
import h5py
import numpy as np
import netCDF4 as nc

# ==================== #
# main function
def main():
    # load data
    pr_list = np.array([0, 10, 20, 30, 40, 50]) # candidate to load

    # design coordinate
    lat = np.linspace(-90, 90, 64)
    lon = np.linspace(0, 360, 129)[:-1]
    time = np.linspace(0, 78000/6-1, 78000)

    cond = np.where((lat>=-15)&(lat<=15))[0]

    print(cond)
    # # load data
    for pr in pr_list:
        with h5py.File(
                f"/home/b11209013/2024_Research/PowerSpec_Dycore/data/PR{pr}_500_20000day_6hourly_prec.dat"
                ) as f:
            prec_data = np.array(f["prec"][:, cond, :])
        
        with nc.Dataset(f"/home/b11209013/2024_Research/PowerSpec_Dycore/data/prec_pr{pr}.nc", "w") as f:
            lon_dim = f.createDimension("lon", 128)
            lat_dim = f.createDimension("lat", 10)
            time_dim = f.createDimension("time", 78000)

            lon_var = f.createVariable("lon", np.float64, ("lon",))
            lon_var[:] = lon

            lat_var = f.createVariable("lat", np.float64, ("lat",))
            lat_var[:] = lat[cond]

            time_var = f.createVariable("time", np.float64, ("time",))
            time_var[:] = time

            prec0 = f.createVariable("prec", np.float64, ("time", "lat", "lon"))
            prec0[:] = prec_data

        print(f"PR {pr} finished")

# ==================== #
# execution section
if __name__ == "__main__":
    main()
