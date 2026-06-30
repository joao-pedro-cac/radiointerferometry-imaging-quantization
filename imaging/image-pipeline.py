import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ducc0.wgridder as wgrid
from time import time
import pfb_imaging.deconv.clark as pfb_clark
import pfb_imaging.deconv.hogbom as pfb_hogbom
from fileread import extract_pixel_info
# from clean import *
from astropy.io import fits
from image_data_analysis import create_histogram


# auxiliary variables
BARSPACE = 80
BARCHAR = '*'
VISIBILITY_MAXNUMBER = 4096
SIMULATED_DATA_PATH = "../../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"
TRUTH_MODEL_PATH = "../../data/simulated/point_field_ctrl/truth_model.fits"


# data set simulated by OSKAR
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_PATH, engine="zarr")
print(oskar_simulated_dataset)


# visibilities data subset
uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]

uvw_data = uvw.data
freq_data = freq.data
vis_data = np.squeeze(vis.data)
weight_data = np.squeeze(weight.data)







for i in range(len(uvw_data)):
    for j in range(len(uvw_data[i])):
        uvw_data[i,j] = int(uvw_data[i,j])



create_histogram(uvw_data, "./images/histogram.png", xlog=False, ylog=True)














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
time_start = time()
model_cube, status = pfb_hogbom.hogbom(
    dirty_image[np.newaxis, :, :],     # new axis to match three dimensions
    psf[np.newaxis, :, :],             # new axis to match three dimensions
    threshold=0.0,                     # force it to rely on the loop fractional threshold (pf)
    gamma=0.001,                       # loop gain
    pf=0.005,                          # stop when max residual drops to (pf * peak_dirty_value)
    maxit=1000,
    verbosity=1
)

# Extract the 2D model
model = np.squeeze(model_cube)
time_stop = time()

print(f"Computation time (clean image): {time_stop - time_start} seconds")

# Extract the 2D model from the 3D model_cube for the plotting code below
model = np.squeeze(model_cube)
print(f"CLEAN status: {"OK" if status == 0 else "ERROR"}")
print(BARCHAR * BARSPACE)













# # =============================================================================
# # CLEAN algorithm (Switching from Hogbom to Clark)
# # =============================================================================
# time_start = time()

# # 1. Prepare 3D representations for dirty image and PSF
# dirty_3d = dirty_image[np.newaxis, :, :]
# psf_3d = psf[np.newaxis, :, :]

# # 2. Generate the mandatory parameters required by Clark
# wsums = np.array([1.0], dtype=dirty_image.dtype)
# mask = np.ones(dirty_image.shape, dtype=dirty_image.dtype)

# # Compute the Fourier transform of the PSF (psfhat)
# # Using rfft2 as is standard for real-valued spatial grid convolutions
# psfhat = np.fft.rfft2(psf_3d, axes=(1, 2))

# print("Running Clark CLEAN...")
# model_cube, status = pfb_clark.clark(
#     dirty=dirty_3d, 
#     psf=psf_3d, 
#     psfhat=psfhat,
#     wsums=wsums,
#     mask=mask,
#     threshold=0.0,            # Force reliance on fractional threshold (pf)
#     gamma=0.05,               # Default loop gain for Clark (can be adjusted)
#     pf=0.001,                 # Stop when max residual drops to 0.1% of peak
#     maxit=50,                 # Major cycles max limit
#     subpf=0.5,                # Subminor loop threshold fraction
#     submaxit=1000,            # Max iterations per subminor loop
#     verbosity=1,
#     nthreads=4                # Match your wgridder thread count
# )

# # Extract the 2D model from the 3D model_cube for the plotting code below
# model = np.squeeze(model_cube)
# time_stop = time()

# print(f"Computation time (Clean image): {time_stop - time_start} seconds")
# print(f"CLEAN status: {'OK' if status == 0 else 'ERROR'}")
# print(BARCHAR * BARSPACE)


























# Define your output filename
output_filename = "reconstructed_clean_model.fits"

try:
    # Open the original truth file to copy its coordinate system (header)
    with fits.open(TRUTH_MODEL_PATH) as hdul:
        original_header = hdul[0].header
        
    # Create a new Primary HDU with your 2D model data and the copied header
    new_hdu = fits.PrimaryHDU(data=model, header=original_header)
    
    # Save the file (overwrite=True prevents errors if you run the script multiple times)
    new_hdu.writeto(output_filename, overwrite=True)
    print(f"Successfully saved CLEAN image to {output_filename}")

except FileNotFoundError:
    # Fallback if truth_model.fits isn't in the working directory
    new_hdu = fits.PrimaryHDU(data=model)
    new_hdu.writeto(output_filename, overwrite=True)
    print(f"Saved CLEAN image to {output_filename} (without header metadata)")

































# plotting results
fig, axes = plt.subplots(figsize=(14, 12))



# model image

percentile = np.percentile(model, 100)
model_new = np.clip(model, -np.nanstd(model), percentile)


im = axes.imshow(model_new, cmap='inferno', origin='lower')
# im = axes.imshow(model, cmap='grey', origin='lower')
plt.colorbar(im, ax=axes, label='Intensity')
axes.set_title('Model Image')

plt.tight_layout()
plt.savefig('images/model.png', format='png')



# dirty image
for p in [90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
    percentile = np.percentile(dirty_image, p)
    dirty_image_new = np.clip(dirty_image, -np.nanstd(dirty_image), percentile)

    im = axes.imshow(dirty_image_new, cmap='inferno', origin='lower')
    axes.set_title(f'Dirty Image (percentile = {p}%)')
    plt.tight_layout()

    if int(p) == p:
        filename = str(p)
    else:
        whole_part = int(p)
        decimal_part = p - whole_part
        filename = str(whole_part) + '_' + str(decimal_part)[2:5]
    plt.savefig(f'images/dirty/dirty-image-{filename}.png', format='png')

