# import packages
using PyCall
using Statistics
using NCDatasets
using LinearAlgebra
using DataStructures

@pyimport numpy as np

path = "/work/b11209013/ERA5_tropic"

# Load data
var_name = ["z", "t", "u", "v", "w"]


# Load dimension
lon = convert(Array{Float64}, Dataset(path*"/z_tropic.nc", "r") do ds
    ds["lon"].var[:]
end)

lat = convert(Array{Float64}, Dataset(path*"/z_tropic.nc", "r") do ds
    ds["lat"].var[:]
end)

time = convert(Array{Float64}, Dataset(path*"/z_tropic.nc", "r") do ds
    ds["time"].var[:]
    end)

# Load variables
var_arr = Array{Float64}(undef, 5, size(lon, 1), size(lat, 1), 8, size(time, 1))

for (i, varn) in enumerate(var_name)
    var=Dataset(path*"/$(varn)_tropic.nc", "r") do ds
        ds["$(varn)"].var[:, :, :, :]
    end

    var_arr[i, :, :, :, :] = convert(Array{Float64}, var)
end

level = Array{Float64}([100000. 92500. 85000. 70000. 50000. 25000. 20000. 10000.])

println("finish loading data")

# DSE com
dse = 1004.5*var_arr[2, :, :, :, :].+var_arr[1, :, :, :, :]

dt = 86400.
dx = 2*pi*6.371e6*cos.(deg2rad.(lat))/576.
dy = 2*pi*6.371e6/720.

dse_dt = similar(dse)
dse_dx = similar(dse)
dse_dy = similar(dse)
dse_dz = similar(dse)
## derivative on time

dse_dt[:, :, :, 1] = (dse[:, :, :, 2]-dse[:, :, :, 1])/dt
dse_dt[:, :, :, 2:end-1] = (dse[:, :, :, 3:end]-dse[:, :, :, 1:end-2])/(2*dt)

dse_dt[:, :, :, end] = (dse[:, :, :, end]-dse[:, :, :, end-1])/(-dt)
## derivative on longitude
for i = 1:length(dx)
    dse_dx[1, i, :, :] = (dse[2, i, :, :]-dse[1, i, :, :])./(dx[i])
    dse_dx[2:end-1, i, :, :] = (dse[3:end, i, :, :]-dse[1:end-2, i, :, :])./(2*dx[i])
    dse_dx[end, i, :, :] = (dse[end, i, :, :]-dse[end-1, i, :, :])/(-dx[i])
end

## derivative on latitude
dse_dy[:, 1, :, :] = (dse[:, 2, :, :]-dse[:, 1, :, :])/dy
dse_dy[:, 2:end-1, :, :] = (dse[:, 3:end, :, :]-dse[:, 1:end-2, :, :])/(2*dy)
dse_dy[:, end, :, :] = (dse[:, end, :, :]-dse[:, end-1, :, :])/(-dy)

## derivative on altitude
dse_dz[:, :, 1, :] = (dse[:, :, 2, :]-dse[:, :, 1, :])/(level[2]-level[1])
dse_dz[:, :, 2:end-1, :] = mapslices(x -> x ./ (level[3:end]-level[1:end-2]), (dse[:, :, 3:end, :]-dse[:, :, 1:end-2, :]), dims=3)
dse_dz[:, :, end, :] = (dse[:, :, end, :]-dse[:, :, end-1, :])/(level[end]-level[end-1])

## Compute Q1
Q1 = dse_dt.+var_arr[3, :, :, :, :].*dse_dx.+var_arr[4, :, :, :, :].*dse_dy.+var_arr[5, :, :, :, :].*dse_dz
println(size(Q1))
println("finish computing Q1")

var_arr = Nothing
dse = Nothing
dse_dt = Nothing
dse_dx = Nothing
dse_dy = Nothing
dse_dz = Nothing

# create netcdf file
ds = Dataset("/work/b11209013/nstcCCKW/Q1.nc", "c")

defDim(ds,"lon", size(lon, 1))
defDim(ds,"lat", size(lat, 1))
defDim(ds, "lev", 8)
defDim(ds, "time", size(time, 1))


lon_d = defVar(ds, "lon", Float64, ("lon", ), attrib=OrderedDict("units" => "degree Eastward"))
lat_d = defVar(ds, "lat", Float64, ("lat", ), attrib=OrderedDict("units" => "degree Northward"))
time_d = defVar(ds, "time", Float64, ("time", ), attrib=OrderedDict("units" => "hours since 1900-01-01 00:00:00.0", "calendar" => "standard"))
lev_d = defVar(ds, "lev", Float64, ("lev", ), attrib=OrderedDict("units" => "hPa"))
q1 = defVar(ds, "Q1", Float64, ("lon", "lat", "lev", "time"), attrib=OrderedDict("units" => "J/kg/s"))

lon_d[:] = lon
lat_d[:] = lat
lev_d[:] = level
time_d[:] = time

q1[:, :, :, :] = Q1
close(ds)
println("finish writing data")
