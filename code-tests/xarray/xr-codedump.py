import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

xr.set_options(display_style="text")

array = np.random.randint(low=0, high=10, size=(2, 3), dtype='u1')

print(array)
print(array.dtype)

data_array = xr.DataArray(array, dims=("x", "y"), coords={"x" : [0, 1], "y" : [0, 1, 2]})
data_array.attrs["description"] = "Simple DataArray object for library testing"

print(data_array)
print()
print(data_array.dtype)
print()
print(data_array.attrs)
print()
print(data_array.data)
print()
print(data_array.dims)
print()
print(data_array.coords)
print()
print(data_array.indexes)
print()

array_rng = np.random.randint(low=0, high=10, size=(5, 8))
print(array_rng)
