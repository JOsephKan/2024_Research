# %% basic setting
# # import package
import os 
import glob
import numpy as np
import netCDF4 as nc

import pyhdf
from pyhdf.HC import HC
from pyhdf.SD import SD, SDC, HDF4Error
from pyhdf.VS import VS
from pyhdf.HDF import HDF

from multiprocessing import Pool

# %% define function
# # load data
def load_data(fname):
    hdf = HDF(fname, SDC.READ)

    vs = hdf.vstart()
    lat = np.stack(vs.attach("Latitude")[:]).reshape(-1)
    lon = np.stack(vs.attach("Longitude")[:]).reshape(-1)
    vs.end()

    for i in range(lon.shape[0]):
        if lon[i] <= 0:
            lon[i] += 360

    cond = np.where(
        (lat >= -15) & (lat <= 15) & (lon >= 160) & (lon <= 260)
    )

    lon = lon[cond]; lat = lat[cond]

    file_sd = SD(fname, SDC.READ)
    # 100.0 is scale factor, check in official website of CloudSat
    hgt = np.array(file_sd.select("Height")[:][cond])
    # 0 of QR stands for shortwave radiation
    qsw = np.array(file_sd.select("QR")[0][cond]) / 100.0
    # 1 of QR stands for longwave radiation
    qlw = np.array(file_sd.select("QR")[1][cond]) / 100.0
    file_sd.end()

    result = {
        "lon": lon, "lat": lat, "hgt": hgt,
        "qlw": qlw, "qsw": qsw
    }

    return result

# %% load complete data
# load date information
dates = np.loadtxt("/home/b11209013/LRF/txt_file/TotalDate.txt", dtype=str)

# load CloudSat data
# # dictrionary for storing data
data = {
    "lon": [], "lat": [], "hgt": [],
    "qlw": [], "qsw": []
}

for d, date in enumerate(dates):
    try:
        # check if the folder exists
        folder = f"/work/DATA/Satellite/CloudSat/{date}/"

        if not os.path.exists(folder):
            print(f"{folder} does not exist")
            continue
        
        file_temp = glob.glob(f"{folder}*.hdf")

        # chunk the data
        with Pool() as p:
            data_temp = p.map(load_data, file_temp)
        
        for i, dt in enumerate(data_temp):
            data["lon"].append(dt["lon"])
            data["lat"].append(dt["lat"])
            data["hgt"].append(dt["hgt"])
            data["qlw"].append(dt["qlw"])
            data["qsw"].append(dt["qsw"])

    except HDF4Error:
        print(f"{date} is not loaded")

    print(f"{date} is loaded")
    
print(len(data["lon"]))