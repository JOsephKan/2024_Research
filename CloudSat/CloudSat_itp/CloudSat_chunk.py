# This program is to interpolate the CloudSat data onto the ERA5 grid
# import package
import os
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc

from numba import jit
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

year = int(sys.argv[1])
date = int(sys.argv[2])

def check_file_exists(filepath):
    if not os.path.exists(filepath):
#         print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code



# ============ #

# load data
## geopotential file path
gpath = f'/work/b11209013/2024_Research/CloudSat/z_itp/{year}.joblib'

## CloudSat file path
cpath = f'/work/b11209013/LRF/data/CloudSat/Daily/{year}_{date}.joblib'

check_file_exists(cpath)

## load gopotential data
z_data = jl.load(gpath)

lon = z_data['lon']
lat = z_data['lat']
lev = z_data['lev']


## load CloudSat data
cloudsat = jl.load(cpath)

# ============== #

# separate the CloudSat data into grid
@jit(nopython=True)
def process_grid_point(data_lon, data_lat, data_hgt, data_qlw, data_qsw, lo, la):
    cond = (
        (data_lon >= lo-0.25) &
        (data_lon <= lo+0.25) &
        (data_lat >= la-0.25) &
        (data_lat <= la+0.25)
    )
    return data_hgt[cond], data_qlw[cond], data_qsw[cond]

def chunking(
        lat_grid, lon_grid,
        data):
    hgt_chunked = np.zeros((lat_grid.size, lon_grid.size), dtype=np.object_)
    qlw_chunked = np.zeros((lat_grid.size, lon_grid.size), dtype=np.object_)
    qsw_chunked = np.zeros((lat_grid.size, lon_grid.size), dtype=np.object_)

    # Convert data arrays to numpy arrays for faster processing
    data_lon = np.array(data['lon'])
    data_lat = np.array(data['lat'])
    data_hgt = np.array(data['hgt'])
    data_qlw = np.array(data['qlw'])
    data_qsw = np.array(data['qsw'])

    # Use numpy broadcasting for faster computation
    for a, la in enumerate(lat_grid):
        for o, lo in enumerate(lon_grid):
            hgt, qlw, qsw = process_grid_point(
                data_lon, data_lat, data_hgt, data_qlw, data_qsw, lo, la
            )
            
            if len(hgt) == 0:
                hgt_chunked[a, o] = []
                qlw_chunked[a, o] = []
                qsw_chunked[a, o] = []
            else:
                hgt_chunked[a, o] = hgt
                qlw_chunked[a, o] = qlw
                qsw_chunked[a, o] = qsw

    return {
        'hgt': hgt_chunked,
        'qlw': qlw_chunked,
        'qsw': qsw_chunked,
    }

chunked_data = Parallel(n_jobs=16)(delayed(chunking)(lat, lon, f) for f in cloudsat)
print((chunked_data[0]['qlw']).shape)
data_merged = {
    'hgt': np.zeros((lat.size, lon.size), dtype=np.object_),
    'qlw': np.zeros((lat.size, lon.size), dtype=np.object_),
    'qsw': np.zeros((lat.size, lon.size), dtype=np.object_),
}

for a, la in enumerate(lat):
    for o, lo in enumerate(lon):
        hgt = []; qlw = []; qsw = []

        for i in range(len(chunked_data)):

            a = int(a); o = int(o):
            hgt.extend(chunked_data[i]['hgt'][a, o])
            qlw.extend(chunked_data[i]['qlw'][a, o])
            qsw.extend(chunked_data[i]['qsw'][a, o])

        data_merged['hgt'][a, o] = hgt
        data_merged['qlw'][a, o] = qlw
        data_merged['qsw'][a, o] = qsw

output_dict = {
    'lon': lon,
    'lat': lat,
    'hgt': data_merged['hgt'],
    'qlw': data_merged['qlw'],
    'qsw': data_merged['qsw']
}

jl.dump(output_dict, f'/work/b11209013/2024_Research/CloudSat/CloudSat_chunked/{year}_{date}.joblib', compress=('zlib', 1))

print(f'{year} {date} output finished')
