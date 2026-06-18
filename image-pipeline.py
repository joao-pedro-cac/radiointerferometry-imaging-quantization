import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import ducc0.wgridder as wgrid
from time import time
import pfb_imaging.deconv.clark as pfb_clark
import pfb_imaging.deconv.hogbom as pfb_hogbom
from clean import hogbom_clean
from shannon import freq_max

# auxilary variables
BARSPACE = 90
VISIBILITY_MAXNUMBER = 4096
IMAGE_SIZE_X = 2520
IMAGE_SIZE_Y = 2520
LOOP_NUMBER = 1
SIMULATED_DATA_ABSOLUTEPATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"


# data set simulated by OSKAR
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_ABSOLUTEPATH, engine="zarr")
print(oskar_simulated_dataset)
print()


# visibilities data subset
uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]

# time_start = time()
# res = freq_max(uvw)
# time_stop = time()

# print(f"\n max(NORMS uv) = {res} ({time_stop - time_start} s)")

uvw_data = uvw.data
freq_data = freq.data
vis_data = np.squeeze(vis.data)
weight_data = np.squeeze(weight.data)




# image resolution parameters
# pixel_size_x = np.radians(1.0) / IMAGE_SIZE_X  # 1 degree projected over N pixels (in projected radians)
# pixel_size_y = np.radians(1.0) / IMAGE_SIZE_Y
# npix_x, npix_y = IMAGE_SIZE_X, IMAGE_SIZE_Y    # image dimensions (in pixels)
# pixel_size_x = np.radians(1.0) / (50 * IMAGE_SIZE_X)
# pixel_size_y = np.radians(1.0) / (50 * IMAGE_SIZE_Y)










# =====================================================================
# Target Metrics Matched to truth_dirty.fits & truth_model.fits
# =====================================================================

# 1. Exact pixel boundaries from FITS header (NAXIS1, NAXIS2)
npix_x = 2520
npix_y = 2520

# 2. Convert CDELT from degrees to radians for ducc0.wgridder
# CDELT = 0.00039890703636168
pixel_size_deg = 0.00039890703636168
pixel_size_x = np.radians(pixel_size_deg)
pixel_size_y = np.radians(pixel_size_deg)

print(f"--- FITS Truth Parameters ---")
print(f"Grid Size:  {npix_x} x {npix_y} pixels")
print(f"Pixel Size: {pixel_size_x} radians ({pixel_size_deg}°)")
print(f"Total FOV:  {np.degrees(pixel_size_x * npix_x):.3f}°")
print(f"-----------------------------\n")








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

print(f"Image dirty shape: {dirty_image.shape}\n")
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

print(f"Computation time (PSF):         {time_stop_psf - time_start_psf} seconds\n")




# Run CLEAN on your existing arrays
# # res = pfb_clark.clark(dirty=dirty_image, psf=psf, psfhat=)
# model, residual, restored, components = hogbom_clean(
#     dirty_image,
#     psf,
#     niter=10000,
#     gain=0.1,
#     threshold=np.std(dirty_image) * 0.005,  # 3 sigma threshold
#     verbose=True
# )


offset_dirty_image = dirty_image + abs(float(np.min(dirty_image)))                # remove negative values
normalized_dirty_image = offset_dirty_image / float(np.max(offset_dirty_image))   # normalize values

# plotting results
fig, axes = plt.subplots(figsize=(14, 12))

im = axes.imshow(dirty_image, cmap='inferno', origin='lower')
axes.set_title('Dirty Image')

plt.colorbar(im, ax=axes, label='Normalized Intensity')

plt.tight_layout()
plt.savefig('dirty-image.png', format='png')
plt.show()






# COMPRENDRE LA LOGIQUE DERRIÈRE LE CHOIX DU NOMBRE DE PIXELS ET DE LA VALEUR EN RADIANS DE LA TAILLE DES PIXELS
# meerkat.tm DIFFÉRENT DE alma.tm