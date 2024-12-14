# using package
using NCDatasets
using Statistics
using Plots

# Load data
path::String = "/work/b11209013/2024_Research/MPAS/PC/CNTL_PC.nc"; # path for the data

data::Dict{String, Array} = Dict()

NCDataset(path, "r") do ds
    for var in keys(ds)
        data[var] = variable(ds, var) # shape: (time, lat, lon, mode)
    end
end

# time averaged
q1pc1 = reshape(mean(data["q1"][:, :, :, 1], dims=1), (size(data["lat"], 1), size(data["lon"], 1)))

p = contourf(data["lon"], data["lat"], q1pc1, c=:RdBu, colorbar=true, title="q1 PC1 time mean")

savefig("q1PC1.png")
