import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ducc0.wgridder as wgrid
from time import time

# auxilary variables
BARSPACE = 90
VISIBILITY_MAXNUMBER = 4096
IMAGE_SIZE_X = 1 << 12
IMAGE_SIZE_Y = 1 << 12
SIMULATED_DATA_ABSOLUTEPATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"


# data set simulated by OSKAR
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_ABSOLUTEPATH, engine="zarr")
# print(oskar_simulated_dataset)


# visibilities data subset
uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]

uvw_data = uvw.data
freq_data = freq.data
vis_data = np.squeeze(vis.data)
weight_data = np.squeeze(weight.data)


# image resolution parameters
npix_x, npix_y = IMAGE_SIZE_X, IMAGE_SIZE_Y    # image dimensions (in pixels)
pixel_size_x = np.radians(1.0) / IMAGE_SIZE_X  # 1 degree projected over N pixels (in projected radians)
pixel_size_y = np.radians(1.0) / IMAGE_SIZE_Y

time_start_dirty = time()

# dirty image computation
dirty_image = wgrid.vis2dirty(
    uvw=uvw_data,                       # uvw coordinates
    freq=freq_data,                     # channel frequencies
    vis=vis_data,                       # visibilities
    wgt=weight_data,                    # weight array
    npix_x=npix_x,                      # no. pixels in the x-axis
    npix_y=npix_y,                      # no. pixels in the y-axis
    pixsize_x=pixel_size_x,             # x-axis pixel size
    pixsize_y=pixel_size_y,             # y-axis pixel size
    epsilon=1e-5,                       # computation accuracy (>= 1e-5)
    do_wgridding=True,                  # perform the full algorithm
    nthreads=4,                         # no. threads used for computation
    double_precision_accumulation=True  # use double precision for computation
)

time_stop_dirty = time()

print(f"Image dirty shape: {dirty_image.shape}\n\n")
print(f"Computation time (dirty image): {time_stop_dirty - time_start_dirty} seconds")




time_start_psf = time()

# PSF computation
psf = wgrid.vis2dirty(
    uvw=uvw_data,                       # uvw coordinates
    freq=freq_data,                     # channel frequencies
    vis=np.ones_like(vis_data),         # visibilities
    wgt=weight_data,                    # weight array
    npix_x=npix_x,                      # no. pixels in the x-axis
    npix_y=npix_y,                      # no. pixels in the y-axis
    pixsize_x=pixel_size_x,             # x-axis pixel size
    pixsize_y=pixel_size_y,             # y-axis pixel size
    epsilon=1e-5,                       # computation accuracy (>= 1e-5)
    do_wgridding=True,                  # perform the full algorithm
    nthreads=4,                         # no. threads used for computation
    double_precision_accumulation=True  # use double precision for computation
)


time_stop_psf = time()

print(f"Computation time (PSF):         {time_stop_psf - time_start_psf} seconds")




# Get dimensions
nrows = uvw_data.shape[0]
npol = 1  # Stokes I only
nchan = freq_data.shape[0] if freq_data.ndim > 0 else 1



# Reshape data for pfb-imaging format
# pfb-imaging expects (row, pol, chan) or (row, chan) structure
uvw_reshaped = uvw_data.reshape(nrows, 3)
vis_reshaped = vis_data.reshape(nrows, 1, 1) if vis_data.ndim == 1 else vis_data.reshape(nrows, -1)
weight_reshaped = weight_data.reshape(nrows, 1, 1) if weight_data.ndim == 1 else weight_data.reshape(nrows, -1)




# --- FIX FOR PFB-IMAGING COMPATIBILITY (MFS) ---

# # Shape: (chan=1, stokes=1, y, x)
# dirty_pfb = dirty_image[np.newaxis, np.newaxis, :, :].astype(np.float32)
# psf_pfb = psf[np.newaxis, np.newaxis, :, :].astype(np.float32)

# # FIX: Use the mean frequency so the coordinate has a length of 1, matching the data
# mean_freq = np.mean(freq_data)
# freq_coords = np.array([mean_freq], dtype=np.float64)
# stokes_coords = np.array(['I']) 

# dds_pfb = xr.Dataset(
#     {
#         'DIRTY': (('chan', 'stokes', 'y', 'x'), dirty_pfb),
#         'PSF': (('chan', 'stokes', 'y', 'x'), psf_pfb),
#     },
#     coords={
#         'chan': freq_coords,
#         'stokes': stokes_coords,
#         'x': np.arange(npix_x),
#         'y': np.arange(npix_y),
#     }
# )

# # Crucial metadata attributes that kclean looks for
# dds_pfb.attrs.update({
#     'cell_rad': pixel_size_x,      
#     'ra': 0.0,
#     'dec': 0.0,
#     'freq_out': float(mean_freq),
#     'nx': npix_x,
#     'ny': npix_y,
# })

# # Save exactly to the filename pattern pfb kclean expects
# dds_pfb.to_netcdf('dirty_image_main.dds')
# print("Created PFB-compliant MFS DDS file: dirty_image_main.dds")


















































































import os
import shutil

# Define target Zarr workspace matching your pfb naming pattern
target_path = 'dirty_image_I_main.dds'

# 1. Housekeep: purge any old monolithic file blocks interfering with the directory model
if os.path.exists(target_path):
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    else:
        os.remove(target_path)

# 2. Extract standard observational metric metadata anchors
mean_freq = np.mean(freq_data) if 'freq_data' in locals() else 1.0e9
freq_coords = np.array([mean_freq], dtype=np.float64)
stokes_coords = np.array(['I'], dtype='<U1') 

# 3. Build a compliant pfb radio data block array mapping (chan, stokes, y, x)
# Variables MUST use uppercase keys 'DIRTY' and 'PSF'
dds_pfb = xr.Dataset(
    {
        'DIRTY': (('chan', 'stokes', 'y', 'x'), dirty_image[np.newaxis, np.newaxis, :, :].astype(np.float32)),
        'PSF': (('chan', 'stokes', 'y', 'x'), psf[np.newaxis, np.newaxis, :, :].astype(np.float32)),
    },
    coords={
        'chan': freq_coords,
        'stokes': stokes_coords,
        'x': np.arange(npix_x),
        'y': np.arange(npix_y),
    }
)

# 4. Inject physical instrument baseline calibration attributes
dds_pfb.attrs.update({
    'cell_rad': pixel_size_x,      
    'ra': 0.0,
    'dec': 0.0,
    'freq_out': float(mean_freq),
    'nx': npix_x,
    'ny': npix_y,
})

# 5. Native export cleanly structure out directory block nodes via Zarr
dds_pfb.to_zarr(target_path)
print(f"Standardized multi-tier Zarr store directory populated at: {target_path}")













plt.figure(figsize=(10, 8))

# Affichage avec interpolation pour éviter les artéfacts
plt.imshow(dirty_image, 
           cmap='inferno',      # Bon choix pour les images radio
           interpolation='none', # 'bilinear' pour un rendu plus lisse
           origin='lower')      # Important : mettre l'origine en bas

plt.colorbar(label='Intensity (Jy/beam)', fraction=0.046, pad=0.04)
plt.title('Dirty Image', fontsize=14)
plt.xlabel('Pixel X', fontsize=12)
plt.ylabel('Pixel Y', fontsize=12)
plt.tight_layout()
plt.show()