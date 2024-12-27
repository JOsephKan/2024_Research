# import pacakges
import numpy as np
import joblib as jl

# load data
data = jl.load("/work/b11209013/2024_Research/CloudSat/CloudSat_itp/2006_163.joblib")

print("Latitude: \n", data["lat"])
print("Longitude: \n", data["lon"])


print(np.shape(data["qlw"]))
