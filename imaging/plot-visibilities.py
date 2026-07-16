import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

BARSPACE = 90
SIMULATED_DATA_ABSOLUTEPATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"

data_array = xr.DataArray(np.random.randn(2, 3), dims=("x","y"), coords={"x" : ["x1", "x2"], "y" : ["y1", "y2", "y3"]})

print(data_array)
print(type(data_array))
print(data_array.dtype)
print('-' * BARSPACE)
# print(data_array[0, 2])
# print('-' * BARSPACE)
# print(data_array.loc["x1"])
# print('-' * BARSPACE)
# print(data_array.isel(x=0))
# print('-' * BARSPACE)
# print(data_array.sel(x="x1"))

xds = xr.open_dataset(SIMULATED_DATA_ABSOLUTEPATH, engine="zarr")

print(xds)
print(type(xds))

print('-' * BARSPACE)

xds_subset = xds.isel(chan=0)

uvw = xds["UVW"].values
# print(xds_subset)
# print(type(xds_subset))

print(xds_subset["l_beam"].shape)
print(xds_subset["m_beam"].shape)
print(xds_subset["chan"].shape)
print(xds_subset["row"].shape)

plt.figure(figsize=(4, 4))
plt.grid(True)
plt.plot(uvw[:,0],uvw[:,1],'.')
# plt.xlim(-8000,8000)
# plt.ylim(-8000,8000)

plt.figure(figsize=(4, 4))
plt.grid(True)
plt.plot(uvw[0:4096,0],uvw[0:4096,1],'.')
# plt.xlim(-8000,8000)
# plt.ylim(-8000,8000)
plt.show()