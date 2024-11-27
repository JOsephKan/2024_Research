# This program is to interpolate ERA5 z information to LRF-format vertical coordinates
# import package
import os
import sys
import numpy as np
import joblib as jl
import netCDF4 as nc

from scipy.interpolate import interp1d

year = int(sys.argv[1])

# ================== #
def interpolate(
    lev_org,
    z_org,
    lev_new
) -> np.ndarray:
    # Interpolate z to new vertical levels
    data_itp = interp1d(lev_org, z_org, kind='linear', axis=0)(lev_new)

    return data_itp

# ================== #
fname: str = f'/work/b11209013/2024_Research/nstcCCKW/z/z_{year}.nc'

with nc.Dataset(fname) as f:
    # Get z data
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    lev = [1000, 925, 850, 700, 500, 250, 200, 100]
    z = f.variables['z'][:]

ltime, llev, llat, llon = z.shape

# permute and reshape data
z_rs = z.transpose((1, 0, 2, 3)).reshape((z.shape[1], -1))

# interpolate
lev_itp = np.linspace(150, 1000, 18)

z_itp = interpolate(lev[::-1], z_rs[::-1], lev_itp)

z_itp_rs = (z_itp.reshape((18, ltime, llat, llon))).transpose((1, 0, 2, 3))

print(f'{year} geopotential reshaped size: ', z_itp_rs.shape)


# ===================== #
# limit the data to (160~260, -15~15)

lon_lim = np.where((lon >= 160) & (lon <= 260))
lat_lim = np.where((lat >= -15) & (lat <= 15))

z_output = z_itp_rs[:, :, lat_lim[0]][:, :, :, lon_lim[0]]

print(f'{year} geopotential output size: ', z_output.shape)

output_dict = {
    'lon': lon[lon_lim[0]],
    'lat': lat[lat_lim[0]],
    'lev': lev_itp,
    'z'  : z_output
}

# save file
jl.dump(output_dict, f'/work/b11209013/2024_Research/CloudSat/z_itp/{year}.joblib', compress=('zlib', 1))

print(f'{year} geopotential done!')

