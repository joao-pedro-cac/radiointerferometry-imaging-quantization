import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ducc0.wgridder as wgrid
from time import time
import pfb_imaging.deconv.clark as pfb_clark
import pfb_imaging.deconv.hogbom as pfb_hogbom
from fileread import extract_pixel_info
from clean import *


# auxiliary variables
BARSPACE = 80
BARCHAR = '*'
VISIBILITY_MAXNUMBER = 4096
SIMULATED_DATA_PATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"
TRUTH_MODEL_PATH = "../data/simulated/point_field_ctrl/truth_model.fits"


# data set simulated by OSKAR
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_PATH, engine="zarr")


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
npix_x, npix_y, pixel_size_x, pixel_size_y = extract_pixel_info(TRUTH_MODEL_PATH)
pixel_size_x = np.radians(pixel_size_x)    # degrees to radians conversion
pixel_size_y = np.radians(pixel_size_y)


print(BARCHAR * BARSPACE)
print("Computing dirty image...")
time_start = time()

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

time_stop = time()
dirty_image = np.transpose(dirty_image)
dirty_image = dirty_image[::-1, :]


print(f"Dirty image shape: {dirty_image.shape}")
print(f"Computation time (dirty image): {time_stop - time_start} seconds")
print(BARCHAR * BARSPACE)


print("Computing PSF...")
time_start = time()

# PSF computation - Double the pixel grid size to handle edge padding in CLEAN
psf = wgrid.vis2dirty(
    uvw=uvw_data,                       
    freq=freq_data,                     
    vis=np.ones_like(vis_data),         
    wgt=weight_data,                    
    npix_x=2 * npix_x,                  # Changed: 2x grid size
    npix_y=2 * npix_y,                  # Changed: 2x grid size
    pixsize_x=pixel_size_x,             
    pixsize_y=pixel_size_y,             
    epsilon=1e-5,                       
    do_wgridding=True,                  
    nthreads=4,                         
    double_precision_accumulation=True  
)
psf = np.transpose(psf)
psf = psf[::-1, :]

time_stop = time()

print(f"PSF shape: {psf.shape}")
print(f"Computation time (PSF):         {time_stop - time_start} seconds")
print(BARCHAR * BARSPACE)




























































# CLEAN algorithm
# Add the new axis as the first dimension [np.newaxis, :, :] to match (nband, nx, ny)
time_start = time()
model_cube, status = pfb_hogbom.hogbom(
    dirty_image[np.newaxis, :, :], 
    psf[np.newaxis, :, :], 
    threshold=0
)
time_stop = time()

print(f"Computation time (Clean image): {time_stop - time_start} seconds")

# Extract the 2D model from the 3D model_cube for the plotting code below
model = np.squeeze(model_cube)
print(f"CLEAN status: {"OK" if status == 0 else "ERROR"}")
print(f"max model = {np.max(model)} at {np.where(model == np.max(model))}")

# res = hogbom_2d(dirty=dirty_image,
#                 psf=psf,
#                 threshold=np.std(dirty_image) * 0.005)
# model, residual, restored, restored_noresidual, components = hogbom_clean(
#     dirty_image,
#     psf,
#     niter=10000,
#     gain=0.5,
#     threshold=np.max(dirty_image) * 0.005,  # 3 sigma threshold
#     verbose=True
# )











































# plotting results
fig, axes = plt.subplots(figsize=(14, 12))



# model image

# percentile = np.percentile(dirty_image, 99.5)
# model_new = np.clip(dirty_image, -np.nanstd(model), percentile)


# im = axes.imshow(model_new, cmap='inferno', origin='lower')
im = axes.imshow(model, cmap='grey', origin='lower')
plt.colorbar(im, ax=axes, label='Intensity')
axes.set_title('Model Image')

plt.tight_layout()
plt.savefig('images/model.png', format='png')



# dirty image
# rms = np.nanstd(dirty_image)            # standard deviation
# for p in [90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
#     percentile = np.percentile(dirty_image, p)
#     dirty_image_new = np.clip(dirty_image, -rms, percentile)

#     im = axes.imshow(dirty_image_new, cmap='inferno', origin='lower')
#     axes.set_title(f'Dirty Image (percentile = {p}%)')
#     plt.tight_layout()

#     if int(p) == p:
#         filename = str(p)
#     else:
#         whole_part = int(p)
#         decimal_part = p - whole_part
#         filename = str(whole_part) + '_' + str(decimal_part)[2:5]
#     plt.savefig(f'images/dirty/dirty-image-{filename}.png', format='png')