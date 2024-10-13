# import packages
using PyCall
using Statistics
using NCDatasets
using LinearAlgebra
using DataStructures

@pyimport numpy as np

# Load data
path = "/work/b11209013/MPAS_tropic/SD15"

var_name = ["zgrid", "theta", "uReconstructZonal", "uReconstructMeridional", "w"]
var_dict = Dict()

for (i, varn) in enumerate(var_name)
    var_dict["$(varn)"]=Dataset(path*"/$(varn)_tropic.nc", "r") do ds
        ds["$(varn)"].var[:, :, :, :]
    end
end

for (i, varn) in enumerate(var_name[1:end-1])
    var_dict["$(varn)"]=var_dict["$(varn)"][:, :, :, 125:end]
end

println(size(var_dict["zgrid"]))

# Load dimension
lon = convert(Array{Float64}, Dataset(path*"/zgrid_tropic.nc", "r") do ds
                  ds["longitude"].var[:]
              end)

lat = convert(Array{Float64}, Dataset(path*"/zgrid_tropic.nc", "r") do ds
                  ds["latitude"].var[:]
              end)

level = convert(Array{Float64}, Dataset(path*"/zgrid_tropic.nc", "r") do ds
                    ds["level"].var[:]
              end)

var_dict["temp"] = var_dict["theta"].*((1000 ./ reshape(level, 1, 1, 38, 1)).^(-287.5/1004.5))

println(size(var_dict["temp"]))

# DSE com
dse = 1004.5*var_dict["temp"].+var_dict["zgrid"]

dt = 86400.
dx = 2*π*6.371e6*cos.(deg2rad.(lat))/576.
dy = 2*π*6.371e6/720.

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
Q1 = dse_dt.+var_dict["uReconstructZonal"].*dse_dx.+var_dict["uReconstructMeridional"].*dse_dy.+var_dict["w"].*dse_dz
println(size(Q1))

ds = Dataset("/work/b11209013/MPAS_tropic/SD15/Q1.nc", "c")

defDim(ds,"lon", 720)
defDim(ds,"lat", 60)
defDim(ds, "level", 38)
defDim(ds, "time", 612)


lon_d = defVar(ds, "lon", Float64, ("lon", ), attrib=OrderedDict("units" => "degree Eastward"))
lat_d = defVar(ds, "lat", Float64, ("lat", ), attrib=OrderedDict("units" => "degree Northward"))
lev_d = defVar(ds, "level", Float64, ("level", ), attrib=OrderedDict("units" => "hPa"))
q1 = defVar(ds, "Q1", Float64, ("lon", "lat", "level", "time"), attrib=OrderedDict("units" => "J/kg/s"))

lon_d[:] = lon
lat_d[:] = lat
lev_d[:] = level

q1[:, :, :, :] = Q1
close(ds)
