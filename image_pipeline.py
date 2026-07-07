import os
import tracemalloc
import numpy as np
import xarray as xr
import ducc0.wgridder as wgrid
import matplotlib.pyplot as plt
import pfb_imaging.deconv.clark as pfb_clark
import pfb_imaging.deconv.hogbom as pfb_hogbom
import quantization.ieee_754_casting as ieee_casting
import imaging.image_data_analysis as image_data_analysis
from imaging.fileread import *
from astropy.io import fits
from time import time, localtime
from json import dump as json_dump
from calendar import month_abbr



# experiment time and day
year, month, day, hour, min, sec, _, _, _ = localtime()
day_dirname = f"results-{day}-{month_abbr[month]}-{year}"
time_dirname = f"experiment_{hour}-{min}-{sec}"
fits_output_filename = "reconstructed_clean_model.fits"


# auxiliary variables
BARSPACE = 80
BARCHAR = '-'
VISIBILITY_MAXNUMBER = 4096
SIMULATED_DATA_PATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"
TRUTH_MODEL_PATH = "../data/simulated/point_field_ctrl/truth_model.fits"
SIMULATION_SCRIPT_PATH = "../scripts/run_sim_ctrl.sh"



# image reconstruction parameters
CLEAN_VARIANT = "hogbom"
GRIDDING_EPSILON = 1e-5
CLEAN_GAMMA = 0.1
CLEAN_PF = 0.5
CLEAN_MAXITER = 100
MAJORLOOP_NUMITER = 10



##############################################################################################################
##############################################################################################################
##############################################################################################################



# data set simulated by OSKAR
telescope_name = extract_telescope_file(SIMULATION_SCRIPT_PATH)
truth_image = np.squeeze(fits.getdata(TRUTH_MODEL_PATH))       # truth (original) sky image
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


# bit encoding of the exponents and tha mantissas for histogram generation
vis_real_exponents = []
vis_real_mantissas = []
vis_imag_exponents = []
vis_imag_mantissas = []

for row in vis_data:
    for elem in row:
        # real part
        sign, exponent, mantissa = ieee_casting.get_double_datafields(elem.real)
        vis_real_exponents.append(exponent.bit_length())
        vis_real_mantissas.append(mantissa.bit_length())

        # imaginary part
        sign, exponent, mantissa = ieee_casting.get_double_datafields(elem.imag)
        vis_imag_exponents.append(exponent.bit_length())
        vis_imag_mantissas.append(mantissa.bit_length())



##############################################################################################################
##############################################################################################################
##############################################################################################################



# image resolution parameters
npix_x, npix_y, pixel_size_x, pixel_size_y = extract_pixel_info(TRUTH_MODEL_PATH)
pixel_size_x = np.radians(pixel_size_x)    # degrees to radians conversion
pixel_size_y = np.radians(pixel_size_y)

print(BARCHAR * BARSPACE)
print("Generating dirty image...")
time_start = time()
tracemalloc.start()

ram_consumption_dirty_image = tracemalloc.get_traced_memory()[0]

# dirty image computation
assert GRIDDING_EPSILON >= 1e-5         # minimum accuracy
dirty_image = wgrid.vis2dirty(
    uvw=uvw_data,                       # uvw coordinates
    freq=freq_data,                     # channel frequencies
    vis=vis_data,                       # visibilities
    wgt=weight_data,                    # weight array_dirty_image
    npix_x=npix_x,                      # no. pixels in the x-axis
    npix_y=npix_y,                      # no. pixels in the y-axis
    pixsize_x=pixel_size_x,             # x-axis pixel size
    pixsize_y=pixel_size_y,             # y-axis pixel size
    epsilon=GRIDDING_EPSILON,           # computation accuracy
    do_wgridding=True,                  # perform the full algorithm
    nthreads=4,                         # no. threads used for computation
    double_precision_accumulation=True, # use double precision for computation
    verbosity=0
)

ram_consumption_dirty_image = tracemalloc.get_traced_memory()[0] - ram_consumption_dirty_image
time_stop = time()
dirtyimage_computing_time = time_stop - time_start


print(f"Dirty image generated")
print(f"Computation time (dirty image): {dirtyimage_computing_time} seconds")
print(BARCHAR * BARSPACE)


# bit encoding of the exponents and tha mantissas for histogram generation
dirty_image_exponents = []
dirty_image_mantissas = []

for row in dirty_image:
    for elem in row:
        sign, exponent, mantissa = ieee_casting.get_double_datafields(elem)
        dirty_image_exponents.append(exponent.bit_length())
        dirty_image_mantissas.append(mantissa.bit_length())



##############################################################################################################
##############################################################################################################
##############################################################################################################



print("Generating PSF...")
time_start = time()
ram_consumption_psf = tracemalloc.get_traced_memory()[0]


# PSF computation
psf = wgrid.vis2dirty(
    uvw=uvw_data,                       
    freq=freq_data,                     
    vis=np.ones_like(vis_data),         
    wgt=weight_data,                    
    npix_x=2 * npix_x,                  # 2x grid size (to handle edge padding in CLEAN)
    npix_y=2 * npix_y,                  # 2x grid size (to handle edge padding in CLEAN)
    pixsize_x=pixel_size_x,             
    pixsize_y=pixel_size_y,             
    epsilon=GRIDDING_EPSILON,           
    do_wgridding=True,                  
    nthreads=4,                         
    double_precision_accumulation=True,
    verbosity=0
)

ram_consumption_psf = tracemalloc.get_traced_memory()[0] - ram_consumption_psf
time_stop = time()
psf_computing_time = time_stop - time_start


print(f"PSF generated")
print(f"Computation time (PSF): {psf_computing_time} seconds")
print(BARCHAR * BARSPACE)


# bit encoding of the exponents and tha mantissas for histogram generation
psf_exponents = []
psf_mantissas = []

for row in psf:
    for elem in row:
        sign, exponent, mantissa = ieee_casting.get_double_datafields(elem)
        psf_exponents.append(exponent.bit_length())
        psf_mantissas.append(mantissa.bit_length())



##############################################################################################################
##############################################################################################################
##############################################################################################################



# CLEAN algorithm
print("Processing clean image via CLEAN...")
time_start = time()
ram_consumption_clean_image = tracemalloc.get_traced_memory()[0]

if CLEAN_VARIANT == "hogbom":
    model_cube, status = pfb_hogbom.hogbom(
        dirty_image[np.newaxis, :, :],     # new axis to match three dimensions
        psf[np.newaxis, :, :],             # new axis to match three dimensions
        threshold=0.0,                     # force it to rely on the loop fractional threshold (pf)
        gamma=CLEAN_GAMMA,                 # loop gain
        pf=CLEAN_PF,                       # stop when max residual drops to (pf * peak_dirty_value)
        maxit=CLEAN_MAXITER,
        verbosity=1
    )
elif CLEAN_VARIANT == "clark":
    # FIX: Normalize the dirty image and PSF to Jy/beam
    psf_peak = np.max(psf)
    dirty_image /= psf_peak
    psf /= psf_peak

    # 1. Prepare 3D representations for dirty image and PSF
    dirty_3d = dirty_image[np.newaxis, :, :]
    psf_3d = psf[np.newaxis, :, :]

    # 2. Generate the mandatory parameters required by Clark
    wsums = np.array([1.0], dtype=dirty_image.dtype)
    mask = np.ones(dirty_image.shape, dtype=dirty_image.dtype)

    # Compute the Fourier transform of the PSF (psfhat)
    # Note: In standard pipelines, the PSF must be shifted to the origin before the FFT
    # to prevent phase ramps during convolution.
    psf_3d_shifted = np.fft.ifftshift(psf_3d, axes=(1, 2))
    psfhat = np.fft.rfft2(psf_3d_shifted, axes=(1, 2))

    model_cube, status = pfb_clark.clark(
        dirty=dirty_3d, 
        psf=psf_3d, 
        psfhat=psfhat,
        wsums=wsums,
        mask=mask,
        threshold=0.0,            # Force reliance on fractional threshold (pf)
        gamma=0.05,               # Default loop gain for Clark (can be adjusted)
        pf=0.001,                 # Stop when max residual drops to 0.1% of peak
        maxit=50,                 # Major cycles max limit
        subpf=0.5,                # Subminor loop threshold fraction
        submaxit=1000,            # Max iterations per subminor loop
        verbosity=1,
        nthreads=4                # Match your wgridder thread count
    )
else:
    raise Exception("Invalid CLEAN algorithm variant")

model = np.squeeze(model_cube)         # extract 2D model



##############################################################################################################
##############################################################################################################
##############################################################################################################



# print("Processing clean image via CLEAN...")

# # FIX: Normalize the dirty image and PSF to Jy/beam
# psf_peak = np.max(psf)
# dirty_image /= psf_peak
# psf /= psf_peak

# # 1. Prepare 3D representations for dirty image and PSF
# dirty_3d = dirty_image[np.newaxis, :, :]
# psf_3d = psf[np.newaxis, :, :]

# # 2. Generate the mandatory parameters required by Clark
# wsums = np.array([1.0], dtype=dirty_image.dtype)
# mask = np.ones(dirty_image.shape, dtype=dirty_image.dtype)

# # Compute the Fourier transform of the PSF (psfhat)
# # Note: In standard pipelines, the PSF must be shifted to the origin before the FFT
# # to prevent phase ramps during convolution.
# psf_3d_shifted = np.fft.ifftshift(psf_3d, axes=(1, 2))
# psfhat = np.fft.rfft2(psf_3d_shifted, axes=(1, 2))

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

# print("")















##############################################################################################################
##############################################################################################################
##############################################################################################################



# pipeline loop
k = 1
recomputed_dirty_image = dirty_image
while k <= MAJORLOOP_NUMITER and status == 1:
    recomputed_vis = wgrid.dirty2vis(
        uvw=uvw_data,                       # uvw coordinates
        freq=freq_data,                     # channel frequencies
        dirty=recomputed_dirty_image,       # dirty image
        wgt=weight_data,                    # weight array
        pixsize_x=pixel_size_x,             # x-axis pixel size
        pixsize_y=pixel_size_y,             # y-axis pixel size
        center_x=0,                         # x-coordinate of the center relative to the phase center
        center_y=0,                         # y-coordinate of the center relative to the phase center
        epsilon=GRIDDING_EPSILON,           # computation accuracy (>= 1e-5)
        do_wgridding=True,                  # perform the full algorithm
        nthreads=4,                         # no. threads used for computation    
        verbosity=0
    )

    recomputed_dirty_image =  wgrid.vis2dirty(
        uvw=uvw_data,
        freq=freq_data,
        vis=recomputed_vis,
        wgt=weight_data,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pixel_size_x,
        pixsize_y=pixel_size_y,
        epsilon=GRIDDING_EPSILON,
        do_wgridding=True,
        nthreads=4,
        double_precision_accumulation=True,
        verbosity=0
    )


    if CLEAN_VARIANT == "hogbom":
        model_cube, status = pfb_hogbom.hogbom(
            recomputed_dirty_image[np.newaxis, :, :],
            psf[np.newaxis, :, :],
            threshold=0.0,
            gamma=CLEAN_GAMMA,
            pf=CLEAN_PF,
            maxit=CLEAN_MAXITER,
            verbosity=1
        )
    elif CLEAN_VARIANT == "clark":
        # FIX: Normalize the dirty image and PSF to Jy/beam
        psf_peak = np.max(psf)
        recomputed_dirty_image /= psf_peak
        psf /= psf_peak

        # 1. Prepare 3D representations for dirty image and PSF
        dirty_3d = recomputed_dirty_image[np.newaxis, :, :]
        psf_3d = psf[np.newaxis, :, :]

        # 2. Generate the mandatory parameters required by Clark
        wsums = np.array([1.0], dtype=recomputed_dirty_image.dtype)
        mask = np.ones(recomputed_dirty_image.shape, dtype=recomputed_dirty_image.dtype)

        # Compute the Fourier transform of the PSF (psfhat)
        # Note: In standard pipelines, the PSF must be shifted to the origin before the FFT
        # to prevent phase ramps during convolution.
        psf_3d_shifted = np.fft.ifftshift(psf_3d, axes=(1, 2))
        psfhat = np.fft.rfft2(psf_3d_shifted, axes=(1, 2))

        model_cube, status = pfb_clark.clark(
            dirty=dirty_3d, 
            psf=psf_3d, 
            psfhat=psfhat,
            wsums=wsums,
            mask=mask,
            threshold=0.0,            # Force reliance on fractional threshold (pf)
            gamma=0.05,               # Default loop gain for Clark (can be adjusted)
            pf=0.001,                 # Stop when max residual drops to 0.1% of peak
            maxit=50,                 # Major cycles max limit
            subpf=0.5,                # Subminor loop threshold fraction
            submaxit=1000,            # Max iterations per subminor loop
            verbosity=1,
            nthreads=4                # Match your wgridder thread count
        )

    model = np.squeeze(model_cube)         # extract 2D model
    k += 1


ram_consumption_clean_image = tracemalloc.get_traced_memory()[0] - ram_consumption_clean_image
time_stop = time()
clean_computing_time = time_stop - time_start

print("Image processed")
print(f"Computation time (CLEAN): {clean_computing_time} seconds")
print(BARCHAR * BARSPACE)



##############################################################################################################
##############################################################################################################
##############################################################################################################





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



##############################################################################################################
##############################################################################################################
##############################################################################################################



# save results and metrics
os.chdir("../results")

try:
    os.mkdir(day_dirname)
    print("New day of experiments: new directory created")
except:
    pass
print(f"Experiment day: {day} {month_abbr[month]} {year}")

os.chdir(day_dirname)
os.mkdir(time_dirname)
os.chdir(time_dirname)
print(f"Experiment began at {hour}:{min}:{sec}")
print(BARSPACE * BARCHAR)


print("Saving CLEAN image to a FITS file...")


dirty_image = np.transpose(dirty_image)[::-1, :]
psf = np.transpose(psf)[::-1, :]
model = np.transpose(model)[::-1, :]


# saving clean image to a FITS file
try:
    # copy the original model's coordinate system (header)
    with fits.open(TRUTH_MODEL_PATH) as hdul:
        original_header = hdul[0].header
        
    # create a new FITS files using the clean image and the copied header
    new_hdu = fits.PrimaryHDU(data=model, header=original_header)
    new_hdu.writeto(fits_output_filename, overwrite=True)
    print(f"Successfully saved CLEAN image to {fits_output_filename}")
except FileNotFoundError:      # exception if TRUTH_MODEL_PATH is not found
    new_hdu = fits.PrimaryHDU(data=model)
    new_hdu.writeto(fits_output_filename, overwrite=True)
    print(f"Saved CLEAN image to {fits_output_filename} (without header metadata)")
print(BARSPACE * BARCHAR)


print("Saving experiment parameters and result metrics...")



parameters = {
    "telescope"       : telescope_name,
    "data set"        : {
        "uvw"     : {
            "type" : str(uvw_data.dtype),
            "size" : uvw_data.size
        },
        "freq"    : {
            "type" : str(freq_data.dtype),
            "size" : freq_data.size
        },
        "vis"     : {
            "type" : str(vis_data.dtype),
            "size" : vis_data.size
        },
        "weights" : {
            "type" : str(weight_data.dtype),
            "size" : weight_data.size
        }
    },
    "daytime"         : {
        "day"    : day,
        "month"  : month_abbr[month],
        "year"   : year,
        "hour"   : hour,
        "minute" : min,
        "second" : sec
    },
    "clean_algorithm" : CLEAN_VARIANT,
    "loop_iterations" : MAJORLOOP_NUMITER,
    "dirty_image"     : {
        "pixelsize_x_radians" : pixel_size_x,
        "pixelsize_y_radians" : pixel_size_y,
        "number_pixels_x"     : npix_x,
        "number_pixels_y"     : npix_y,
        "epsilon"             : GRIDDING_EPSILON
    },
    "clean_image"     : {
        "gamma"          : CLEAN_GAMMA,
        "peak_fraction"  : CLEAN_PF,
        "max_iterations" : CLEAN_MAXITER
    }
}

with open("parameters.json", "w") as fd:
    json_dump(parameters, fd)



metrics = {
    "computation_time_seconds"     : {
        "dirty_image" : dirtyimage_computing_time,
        "psf"         : psf_computing_time,
        "clean_image" : clean_computing_time,
        "total"       : dirtyimage_computing_time + psf_computing_time + clean_computing_time
    },
    "memory_consumption_megabytes" : {
        "dirty_image" : ram_consumption_dirty_image / (1024 ** 2),
        "psf"         : ram_consumption_psf / (1024 ** 2),
        "clean_image" : ram_consumption_clean_image / (1024 ** 2),
        "total"       : (ram_consumption_dirty_image + ram_consumption_psf + ram_consumption_clean_image) / (1024 ** 2)
    },
    "rms"                          : image_data_analysis.compute_rms(model),
    "dynamic_range"                : image_data_analysis.compute_dr(model),
    "snr"                          : image_data_analysis.compute_snr(model, truth_image),
    "psnr"                         : image_data_analysis.compute_psnr(model, truth_image),
    "ssim"                         : image_data_analysis.compute_ssim(model, truth_image)
}

with open("metrics.json", "w") as fd:
    json_dump(metrics, fd)

print("Parameters and metrics saved")
print(BARSPACE * BARCHAR)



# plotting results
os.mkdir("images")
os.chdir("images")
os.mkdir("dirty")
print("Plotting results...")



image_data_analysis.create_histogram(dirty_image_exponents, "histogram-dirty_image-exponents.png")
image_data_analysis.create_histogram(dirty_image_mantissas, "histogram-dirty_image-mantissas.png")

image_data_analysis.create_histogram(psf_exponents, "histogram-psf-exponents.png")
image_data_analysis.create_histogram(psf_mantissas, "histogram-psf-mantissas.png")

image_data_analysis.create_histogram(vis_real_exponents, "histogram-visibilities-real_part-exponents.png")
image_data_analysis.create_histogram(vis_real_mantissas, "histogram-visibilities-real_part-mantissas.png")

image_data_analysis.create_histogram(vis_imag_exponents, "histogram-visibilities-imaginary_part-exponents.png")
image_data_analysis.create_histogram(vis_imag_mantissas, "histogram-visibilities-imaginary_part-mantissas.png")

fig, axes = plt.subplots(figsize=(14, 12))


# clean image
im = axes.imshow(model, cmap='inferno', origin='lower')
plt.colorbar(im, ax=axes, label='Intensity')
axes.set_title('Clean Image')

plt.tight_layout()
plt.savefig('clean.png', format='png')


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
    plt.savefig(f'dirty/dirty-image-{filename}.png', format='png')

print("Results plotted and saved to PNG files")
