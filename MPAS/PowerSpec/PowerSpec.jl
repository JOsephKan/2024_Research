# This program is to compute the power spectrum of MPAS data
# Load packages
using Glob
using FFTW
using NCDatasets

# Load data
exp_list:: Array{String, 1} = ["CNTL", "NCRF", "NSC"]

path :: String = "/work/b11209013/2024_Research/MPAS/PC/"

data :: Dict{String, Dict} = Dict()

## Load PC data
for exp in exp_list
  local fname::String = path*"$(exp)_PC.nc"

  data[exp] = Dict()

  NCDataset(fname, "r") do ds
    for var in ["t", "q1"]
      data[exp][var] = variable(ds, var) # shape: (time, lat, lon, mode)
    end
  end
end

## Load EOF data
NCDataset("/work/b11209013/2024_Research/MPAS/PC/EOF.nc", "r") do ds 
  lev = variable(ds, "lev")
end

println("Data shape: ", data["CNTL"]["t"][:])
