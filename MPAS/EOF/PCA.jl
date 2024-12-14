# This program is to compute the PCs of different variables
# %% section 1.
# import package
using NCDatasets
using Statistics
using DataStructures
using LinearAlgebra

# Load data
## information of data files
path::String = "/work/b11209013/2024_Research/MPAS/merged_data/"; # path for the data

exp ::String = "NSC"; # experiment name

var_list::Array{String, 1} = [
    "q1", "qv", "theta",
    "rqvcuten", "rthcuten",
    "rthratenlw", "rthratensw"
]; # variable list

## load latitude
ds = Dataset(joinpath(path, "$(exp)/q1.nc"))

lat  = ds["lat"][:]
lon  = ds["lon"][:]
lev  = ds["lev"][:]
time = ds["time"][:]

close(ds)

lat_lim = findall(x -> x >= -5 && x <= 5, lat)
lat = lat[lat_lim]

## load variables
var_dict = Dict{String, Array}()

for var in var_list
    file_name = joinpath(path, "$(exp)/$(var).nc")

    local ds = Dataset(file_name)

    var_dict[var] = permutedims(ds[var][:, lat_lim, :, :], (3, 4, 2, 1))

    close(ds)
end

llev, ltime, llat, llon = size(var_dict[var_list[1]])

## load eof data
eof_ds = Dataset("/work/b11209013/2024_Research/MPAS/PC/EOF.nc")

EOF = (eof_ds["EOF"][:, :])

close(eof_ds)


# reshape data
var_rs = Dict{String, Array}()

for var in var_list
    var_rs[var] = reshape(var_dict[var], (llev, ltime*llat*llon))'
end

# convert units
theta2t = (1000 ./ lev[1, :]) .^ (-0.286)

var_conv = Dict{String, Array}()

var_conv["q1"]       = var_rs["q1"]*86400/1004.5                # unit: K/day
var_conv["qv"]       = var_rs["qv"]*1000                        # unit: g/kg
var_conv["t"]        = var_rs["theta"].*theta2t                 # unit: K
var_conv["rqvcuten"] = var_rs["rqvcuten"]*86400*1000            # unit: g/kg/day
var_conv["rtcuten"]  = var_rs["rthcuten"] .* theta2t .* 86400   # unit: K/day
var_conv["rtatenlw"] = var_rs["rthratenlw"] .* theta2t .* 86400 # unit: K/day
var_conv["rtatensw"] = var_rs["rthratensw"] .* theta2t .* 86400 # unit: K/day

## compute anomlies
for var in keys(var_conv)
    var_conv[var] = var_conv[var] .- mean(var_conv[var], dims=1)
end

## generate PCs of variables
PC = Dict{String, Array}()

for var in keys(var_conv)
    PC[var] = reshape(var_conv[var] * EOF, (ltime, llat, llon, 10))
end

## Save files
NCDataset("/work/b11209013/2024_Research/MPAS/PC/$(exp)_PC.nc", "c") do ds

    # Step 1: Define dimensions
    defDim(ds, "time", ltime)
    defDim(ds, "lat", llat)
    defDim(ds, "lon", llon)
    defDim(ds, "mode_number", 10)  # Renamed to a valid dimension name

    # Step 2: Define variables
    lat_v = defVar(ds, "lat", Float64, ("lat",), attrib=Dict("units"=>"degree_north", "standard_name"=>"latitude", "long_name"=>"latitude", "axis"=>"Y"))
    lon_v = defVar(ds, "lon", Float64, ("lon",), attrib=Dict("units"=>"degree_east", "standard_name"=>"longitude", "long_name"=>"longitude", "axis"=>"X"))
    time_v = defVar(ds, "time", Float64, ("time",), attrib=Dict("units"=>"hours since 1900-01-01 00:00:00.0", "standard_name"=>"time", "calendar"=>"standard", "axis"=>"T"))
    mode_v = defVar(ds, "mode_number", Int64, ("mode_number",), attrib=Dict("units"=>"None"))

    # Define variables for var_list
    for var in keys(var_conv)
        var_v = defVar(ds, var, Float64, ("time", "lat", "lon", "mode_number"), attrib=Dict("long_name"=>"$(var) PCs"))
        var_v[:,:,:,:] = PC[var]
    end

    # Step 3: Write data to variables
    lat_v[:] = lat
    lon_v[:] = lon
    time_v[:] = time
    mode_v[:] = 1:10

end

