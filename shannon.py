import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr
# import ducc0.wgridder as wgrid
# from time import time
# from clean import hogbom_clean

# # auxilary variables
# BARSPACE = 90
# VISIBILITY_MAXNUMBER = 4096
# IMAGE_SIZE_X = 1 << 6
# IMAGE_SIZE_Y = 1 << 6
# LOOP_NUMBER = 1
# SIMULATED_DATA_ABSOLUTEPATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"


# # data set simulated by OSKAR
# oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_ABSOLUTEPATH, engine="zarr")
# print(oskar_simulated_dataset)
# print()


# # visibilities data subset
# uvw    = oskar_simulated_dataset["UVW"][:VISIBILITY_MAXNUMBER, :2]
# uvw_data = uvw.data

# print(uvw_data.shape)
# print(uvw[:, 0])
# print(uvw[:, 1])

# norms = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2)
# norms = sorted(norms.data)

# print(norms)

# print(f"MAX NORM uv = {float(max(norms))}")

def freq_max(uv):
    norms = np.sqrt(uv[:, 0]**2 + uv[:, 1]**2)
    return float(max(norms))
    



# plt.figure(figsize=(4, 4))
# plt.grid(True)
# plt.plot(uvw[:, 0], uvw[:, 1], '.')

# plt.figure(figsize=(4, 4))
# plt.grid(True)
# plt.plot(norms[:len(norms)//2], '.')
# plt.show()