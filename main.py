import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import quantization.ieee_754_casting as ieee_casting
import imaging.image_data_analysis as image_data_analysis
from imaging.fileread import *
from astropy.io import fits
from calendar import month_abbr
from time import time, localtime
from json import dump as json_dump
from pipeline import ImagePipeline



# experiment time and day
year, month, day, hour, min, sec, _, _, _ = localtime()
day_dirname = f"results-{day}-{month_abbr[month]}-{year}"
time_dirname = f"experiment_{hour}-{min}-{sec}"
fits_output_filename = "reconstructed_clean_model.fits"


# auxiliary variables
BARLENGTH = 80
BARCHAR = '-'
VISIBILITY_MAXNUMBER = 4096
SIMULATED_DATA_PATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"
TRUTH_MODEL_PATH = "../data/simulated/point_field_ctrl/truth_model.fits"
SIMULATION_SCRIPT_PATH = "../scripts/run_sim_ctrl.sh"



# image reconstruction parameters
CLEAN_VARIANT = "hogbom"
GRIDDING_EPSILON = 1e-5
CLEAN_GAMMA = 0.01
CLEAN_PF = 0.1
CLEAN_MAXITER = 25
MAJORLOOP_NUMITER = 5
experiment_commentary = "Comparing float64 x float32 x float16 x bfloat16 using four pipelines and recreating the original sky image." \
                        " Testing pf adaptation so that we always consider the initial dirty image's peak"



########################################################################################################################
########################################################################################################################
########################################################################################################################



# data set simulated by OSKAR
telescope_name = extract_telescope_file(SIMULATION_SCRIPT_PATH)
truth_image = np.squeeze(fits.getdata(TRUTH_MODEL_PATH))       # truth (original) sky image
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_PATH, engine="zarr")
# print(oskar_simulated_dataset)


# visibilities data subset
uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]

uvw_data =    uvw.data
freq_data =   freq.data
vis_data =    np.squeeze(vis.data)
weight_data = np.squeeze(weight.data)



########################################################################################################################
########################################################################################################################
########################################################################################################################



pipeline = ImagePipeline(clean_algorithm=CLEAN_VARIANT,
                         gridding_epsilon=GRIDDING_EPSILON,
                         clean_gamma=CLEAN_GAMMA,
                         clean_pf=CLEAN_PF,
                         clean_maxiter=CLEAN_MAXITER)

pipeline.set_truth_model(TRUTH_MODEL_PATH)



########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float64 image pipeline ################################################

# visibilities type casting
vis_128 = vis_data.astype("complex128")
wgt_64 = weight_data.astype("float64")




#  dirty image generation
print("Generating float64 dirty image...")
dirty_image_64 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_128,
                                              wgt=wgt_64,
                                              verbosity=False)

mem_dirty_image_64            = pipeline.used_mem_dirty_image
peak_mem_dirty_image_64       = pipeline.peak_mem_dirty_image
dirty_image_computing_time_64 = pipeline.computing_time_dirty_image

print("float64 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_64} s")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float64 PSF...")
psf_64 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_128,
                              wgt=wgt_64,
                              verbosity=False)

mem_psf_64            = pipeline.used_mem_psf
peak_mem_psf_64       = pipeline.peak_mem_psf
psf_computing_time_64 = pipeline.computing_time_psf

print("float64 PSF generated")
print(f"Computing time: {psf_computing_time_64} s")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float64 clean model...")
clean_model_64, status_64 = pipeline.compute_clean_image(dirty_image_64, psf_64)
mem_clean_image_64            = pipeline.used_mem_clean_image
peak_mem_clean_image_64       = pipeline.peak_mem_clean_image
clean_image_computing_time_64 = pipeline.computing_time_clean_image

k = 1
recomputed_dirty_image = dirty_image_64

# pipeline loop
while k <= MAJORLOOP_NUMITER and status_64 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_64,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_64)
    recomputed_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                          freq=freq_data,
                                                          vis=recomputed_vis,
                                                          wgt=wgt_64)
    clean_model_64, status_64 = pipeline.compute_clean_image(dirty_image=recomputed_dirty_image,
                                                             psf=psf_64 / np.max(recomputed_dirty_image) * np.max(dirty_image_64))

    clean_image_computing_time_64 += pipeline.computing_time_vis + pipeline.computing_time_dirty_image + pipeline.computing_time_clean_image
    mem_clean_image_64            += pipeline.used_mem_dirty_image + pipeline.used_mem_dirty_image + pipeline.used_mem_clean_image
    peak_mem_clean_image_64        = max(pipeline.peak_mem_dirty_image, pipeline.peak_mem_dirty_image, pipeline.peak_mem_clean_image, peak_mem_clean_image_64)

    k += 1

print("float64 clean model generated")
print(f"Computing time: {clean_image_computing_time_64} s")
print(BARCHAR * BARLENGTH)



########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float32 image pipeline ################################################

# visibilities type casting
vis_64 = vis_data.astype("complex64")
wgt_32 = weight_data.astype("float32")




#  dirty image generation
print("Generating float32 dirty image...")
dirty_image_32 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_64,
                                              wgt=wgt_32,
                                              verbosity=False)

mem_dirty_image_32            = pipeline.used_mem_dirty_image
peak_mem_dirty_image_32       = pipeline.peak_mem_dirty_image
dirty_image_computing_time_32 = pipeline.computing_time_dirty_image

print("float32 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_32} s")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float32 PSF...")
psf_32 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_64,
                              wgt=wgt_32,
                              verbosity=False)

mem_psf_32            = pipeline.used_mem_psf
peak_mem_psf_32       = pipeline.peak_mem_psf
psf_computing_time_32 = pipeline.computing_time_psf

print("float32 PSF generated")
print(f"Computing time: {psf_computing_time_32} s")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float32 clean model...")
clean_model_32, status_32 = pipeline.compute_clean_image(dirty_image_32, psf_32)
mem_clean_image_32            = pipeline.used_mem_clean_image
peak_mem_clean_image_32       = pipeline.peak_mem_clean_image
clean_image_computing_time_32 = pipeline.computing_time_clean_image

k = 1
recomputed_dirty_image = dirty_image_32

# pipeline loop
while k <= MAJORLOOP_NUMITER and status_32 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_32,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_32)
    recomputed_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                          freq=freq_data,
                                                          vis=recomputed_vis,
                                                          wgt=wgt_32)
    clean_model_32, status_32 = pipeline.compute_clean_image(dirty_image=recomputed_dirty_image,
                                                             psf=psf_32 / np.max(recomputed_dirty_image) * np.max(dirty_image_32))

    clean_image_computing_time_32 += pipeline.computing_time_vis + pipeline.computing_time_dirty_image + pipeline.computing_time_clean_image
    mem_clean_image_32            += pipeline.used_mem_dirty_image + pipeline.used_mem_dirty_image + pipeline.used_mem_clean_image
    peak_mem_clean_image_32        = max(pipeline.peak_mem_dirty_image, pipeline.peak_mem_dirty_image, pipeline.peak_mem_clean_image, peak_mem_clean_image_32)

    k += 1

print("float32 clean model generated")
print(f"Computing time: {clean_image_computing_time_32} s")
print(BARCHAR * BARLENGTH)




########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float16 image pipeline ################################################

# visibilities type casting
vis_32 = vis_64.copy()
for i in range(vis_32.shape[0]):
    for j in range(vis_32.shape[1]):
        data = complex(vis_32[i, j])
        vis_32[i, j] = ieee_casting.float_to_half(data.real)[0] + ieee_casting.float_to_half(data.imag)[0] * 1j
wgt_16 = wgt_32.copy()
for i in range(wgt_16.shape[0]):
    for j in range(wgt_16.shape[1]):
        wgt_16[i, j] = ieee_casting.float_to_half(wgt_16[i, j])[0]




#  dirty image generation
print("Generating float16 dirty image...")
dirty_image_16 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_32,
                                              wgt=wgt_16,
                                              verbosity=False)

mem_dirty_image_16            = pipeline.used_mem_dirty_image
peak_mem_dirty_image_16       = pipeline.peak_mem_dirty_image
dirty_image_computing_time_16 = pipeline.computing_time_dirty_image

print("float16 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_16} s")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float16 PSF...")
psf_16 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_32,
                              wgt=wgt_16,
                              verbosity=False)

mem_psf_16            = pipeline.used_mem_psf
peak_mem_psf_16       = pipeline.peak_mem_psf
psf_computing_time_16 = pipeline.computing_time_psf

print("float16 PSF generated")
print(f"Computing time: {psf_computing_time_16} s")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float16 clean model...")
clean_model_16, status_16 = pipeline.compute_clean_image(dirty_image_16, psf_16)
mem_clean_image_16            = pipeline.used_mem_clean_image
peak_mem_clean_image_16       = pipeline.peak_mem_clean_image
clean_image_computing_time_16 = pipeline.computing_time_clean_image

k = 1
recomputed_dirty_image = dirty_image_16

# pipeline loop
while k <= MAJORLOOP_NUMITER and status_16 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_16,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_16)
    recomputed_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                          freq=freq_data,
                                                          vis=recomputed_vis,
                                                          wgt=wgt_16)
    clean_model_16, status_16 = pipeline.compute_clean_image(dirty_image=recomputed_dirty_image,
                                                             psf=psf_16 / np.max(recomputed_dirty_image) * np.max(dirty_image_16))

    clean_image_computing_time_16 += pipeline.computing_time_vis + pipeline.computing_time_dirty_image + pipeline.computing_time_clean_image
    mem_clean_image_16            += pipeline.used_mem_dirty_image + pipeline.used_mem_dirty_image + pipeline.used_mem_clean_image
    peak_mem_clean_image_16        = max(pipeline.peak_mem_dirty_image, pipeline.peak_mem_dirty_image, pipeline.peak_mem_clean_image, peak_mem_clean_image_16)

    k += 1

print("float16 clean model generated")
print(f"Computing time: {clean_image_computing_time_16} s")
print(BARCHAR * BARLENGTH)




########################################################################################################################
########################################################################################################################
########################################################################################################################

############################################# brain float16 image pipeline #############################################

# visibilities type casting
vis_b32 = vis_64.copy()
for i in range(vis_b32.shape[0]):
    for j in range(vis_b32.shape[1]):
        data = complex(vis_b32[i, j])
        vis_b32[i, j] = ieee_casting.float_to_bfloat(data.real)[0] + ieee_casting.float_to_bfloat(data.imag)[0] * 1j
wgt_b16 = wgt_32.copy()
for i in range(wgt_b16.shape[0]):
    for j in range(wgt_b16.shape[1]):
        wgt_b16[i, j] = ieee_casting.float_to_bfloat(wgt_16[i, j])[0]




#  dirty image generation
print("Generating bfloat16 dirty image...")
dirty_image_b16 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_b32,
                                              wgt=wgt_b16,
                                              verbosity=False)

mem_dirty_image_b16            = pipeline.used_mem_dirty_image
peak_mem_dirty_image_b16       = pipeline.peak_mem_dirty_image
dirty_image_computing_time_b16 = pipeline.computing_time_dirty_image

print("bfloat16 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_b16} s")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating bfloat16 PSF...")
psf_b16 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_b32,
                              wgt=wgt_b16,
                              verbosity=False)

mem_psf_b16            = pipeline.used_mem_psf
peak_mem_psf_b16       = pipeline.peak_mem_psf
psf_computing_time_b16 = pipeline.computing_time_psf

print("bfloat16 PSF generated")
print(f"Computing time: {psf_computing_time_b16} s")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating bfloat16 clean model...")
clean_model_b16, status_b16 = pipeline.compute_clean_image(dirty_image_b16, psf_b16)
mem_clean_image_b16            = pipeline.used_mem_clean_image
peak_mem_clean_image_b16       = pipeline.peak_mem_clean_image
clean_image_computing_time_b16 = pipeline.computing_time_clean_image

k = 1
recomputed_dirty_image = dirty_image_b16

# pipeline loop
while k <= MAJORLOOP_NUMITER and status_b16 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_b16,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_b16)
    recomputed_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                          freq=freq_data,
                                                          vis=recomputed_vis,
                                                          wgt=wgt_b16)
    clean_model_b16, status_b16 = pipeline.compute_clean_image(dirty_image=recomputed_dirty_image,
                                                             psf=psf_b16 / np.max(recomputed_dirty_image) * np.max(dirty_image_b16))

    clean_image_computing_time_b16 += pipeline.computing_time_vis + pipeline.computing_time_dirty_image + pipeline.computing_time_clean_image
    mem_clean_image_b16            += pipeline.used_mem_dirty_image + pipeline.used_mem_dirty_image + pipeline.used_mem_clean_image
    peak_mem_clean_image_b16        = max(pipeline.peak_mem_dirty_image, pipeline.peak_mem_dirty_image, pipeline.peak_mem_clean_image, peak_mem_clean_image_b16)

    k += 1

print("bfloat16 clean model generated")
print(f"Computing time: {clean_image_computing_time_b16} s")
print(BARCHAR * BARLENGTH)




########################################################################################################################
########################################################################################################################
########################################################################################################################



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
print(BARLENGTH * BARCHAR)


print("Saving CLEAN images to FITS files...")


dirty_image_64  = np.transpose(dirty_image_64)[::-1, :]
dirty_image_32  = np.transpose(dirty_image_32)[::-1, :]
dirty_image_16  = np.transpose(dirty_image_16)[::-1, :]
dirty_image_b16 = np.transpose(dirty_image_b16)[::-1, :]

# psf = np.transpose(psf)[::-1, :]

clean_model_64  = np.transpose(clean_model_64)[::-1, :]
clean_model_32  = np.transpose(clean_model_32)[::-1, :]
clean_model_16  = np.transpose(clean_model_16)[::-1, :]
clean_model_b16 = np.transpose(clean_model_b16)[::-1, :]


# saving clean image to a FITS file
try:
    # copy the original model's coordinate system (header)
    with fits.open(TRUTH_MODEL_PATH) as hdul:
        original_header = hdul[0].header
        
    # create a new FITS files using the clean image and the copied header
    new_hdu = fits.PrimaryHDU(data=clean_model_64, header=original_header)
    new_hdu.writeto("clean_model_float64", overwrite=True)


    # the same for the 32-bit image
    new_hdu = fits.PrimaryHDU(data=clean_model_32, header=original_header)
    new_hdu.writeto("clean_model_float32", overwrite=True)


    # the same for the 16-bit image
    new_hdu = fits.PrimaryHDU(data=clean_model_16, header=original_header)
    new_hdu.writeto("clean_model_float16", overwrite=True)


    # the same for the brain float image
    new_hdu = fits.PrimaryHDU(data=clean_model_b16, header=original_header)
    new_hdu.writeto("clean_model_bfloat16", overwrite=True)


    print(f"Successfully saved CLEAN images")
except FileNotFoundError:      # exception if TRUTH_MODEL_PATH is not found
    new_hdu = fits.PrimaryHDU(data=clean_model_64)
    new_hdu.writeto("clean_model_float64", overwrite=True)

    new_hdu = fits.PrimaryHDU(data=clean_model_32)
    new_hdu.writeto("clean_model_float32", overwrite=True)

    new_hdu = fits.PrimaryHDU(data=clean_model_16)
    new_hdu.writeto("clean_model_float16", overwrite=True)

    new_hdu = fits.PrimaryHDU(data=clean_model_b16)
    new_hdu.writeto("clean_model_bfloat16", overwrite=True)
    print(f"Saved CLEAN images (without header metadata)")
print(BARLENGTH * BARCHAR)


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
        "epsilon"             : GRIDDING_EPSILON
    },
    "clean_image"     : {
        "gamma"          : CLEAN_GAMMA,
        "peak_fraction"  : CLEAN_PF,
        "max_iterations" : CLEAN_MAXITER
    },
    "commentary"      : experiment_commentary
}

with open("parameters.json", "w") as fd:
    json_dump(parameters, fd)



metrics = {
    "computation_time_seconds"     : {
        "float64"  : {
            "dirty_image" : dirty_image_computing_time_64,
            "psf"         : psf_computing_time_64,
            "clean_image" : clean_image_computing_time_64,
            "total"       : dirty_image_computing_time_64 + psf_computing_time_64 + clean_image_computing_time_64
        },
        "float32"  : {
            "dirty_image" : dirty_image_computing_time_32,
            "psf"         : psf_computing_time_32,
            "clean_image" : clean_image_computing_time_32,
            "total"       : dirty_image_computing_time_32 + psf_computing_time_32 + clean_image_computing_time_32
        },
        "float16"  : {
            "dirty_image" : dirty_image_computing_time_16,
            "psf"         : psf_computing_time_16,
            "clean_image" : clean_image_computing_time_16,
            "total"       : dirty_image_computing_time_16 + psf_computing_time_16 + clean_image_computing_time_16
        },
        "bfloat16" : {
            "dirty_image" : dirty_image_computing_time_b16,
            "psf"         : psf_computing_time_b16,
            "clean_image" : clean_image_computing_time_b16,
            "total"       : dirty_image_computing_time_b16 + psf_computing_time_b16 + clean_image_computing_time_b16
        }
    },
    "memory_consumption_megabytes" : {
        "float64"  : {
            "dirty_image" : {
                "used" : mem_dirty_image_64 / (1024 ** 2),
                "peak" : peak_mem_dirty_image_64 / (1024 ** 2)
            },
            "psf"         : {
                "used" : mem_psf_64 / (1024 ** 2),
                "peak" : peak_mem_psf_64 / (1024 ** 2)
            },
            "clean_image" : {
                "used" : mem_clean_image_64 / (1024 ** 2),
                "peak" : peak_mem_clean_image_64 / (1024 ** 2)
            },
            "total"       : {
                "used" : (mem_dirty_image_64 + mem_psf_64 + mem_clean_image_64) / (1024 ** 2),
                "peak" : (peak_mem_dirty_image_64 + peak_mem_psf_64 + peak_mem_clean_image_64) / (1024 ** 2)
            },
        },
        "float32"  : {
            "dirty_image" : {
                "used" : mem_dirty_image_32 / (1024 ** 2),
                "peak" : peak_mem_dirty_image_32 / (1024 ** 2)
            },
            "psf"         : {
                "used" : mem_psf_32 / (1024 ** 2),
                "peak" : peak_mem_psf_32 / (1024 ** 2)
            },
            "clean_image" : {
                "used" : mem_clean_image_32 / (1024 ** 2),
                "peak" : peak_mem_clean_image_32 / (1024 ** 2)
            },
            "total"       : {
                "used" : (mem_dirty_image_32 + mem_psf_32 + mem_clean_image_32) / (1024 ** 2),
                "peak" : (peak_mem_dirty_image_32 + peak_mem_psf_32 + peak_mem_clean_image_32) / (1024 ** 2)
            },
        },
        "float16"  : {
            "dirty_image" : {
                "used" : mem_dirty_image_16 / (1024 ** 2),
                "peak" : peak_mem_dirty_image_16 / (1024 ** 2)
            },
            "psf"         : {
                "used" : mem_psf_16 / (1024 ** 2),
                "peak" : peak_mem_psf_16 / (1024 ** 2)
            },
            "clean_image" : {
                "used" : mem_clean_image_16 / (1024 ** 2),
                "peak" : peak_mem_clean_image_16 / (1024 ** 2)
            },
            "total"       : {
                "used" : (mem_dirty_image_16 + mem_psf_16 + mem_clean_image_16) / (1024 ** 2),
                "peak" : (peak_mem_dirty_image_16 + peak_mem_psf_16 + peak_mem_clean_image_16) / (1024 ** 2)
            },
        },
        "bfloat16" : {
            "dirty_image" : {
                "used" : mem_dirty_image_b16 / (1024 ** 2),
                "peak" : peak_mem_dirty_image_b16 / (1024 ** 2)
            },
            "psf"         : {
                "used" : mem_psf_b16 / (1024 ** 2),
                "peak" : peak_mem_psf_b16 / (1024 ** 2)
            },
            "clean_image" : {
                "used" : mem_clean_image_b16 / (1024 ** 2),
                "peak" : peak_mem_clean_image_b16 / (1024 ** 2)
            },
            "total"       : {
                "used" : (mem_dirty_image_b16 + mem_psf_b16 + mem_clean_image_b16) / (1024 ** 2),
                "peak" : (peak_mem_dirty_image_b16 + peak_mem_psf_b16 + peak_mem_clean_image_b16) / (1024 ** 2)
            },
        }
    }
}

metrics_64  = pipeline.calculate_image_metrics(clean_model_64)
metrics_32  = pipeline.calculate_image_metrics(clean_model_32)
metrics_16  = pipeline.calculate_image_metrics(clean_model_16)
metrics_b16 = pipeline.calculate_image_metrics(clean_model_b16)

metrics.update({
    "image_metrics" : {
        "float64"  : metrics_64,
        "float32"  : metrics_32,
        "float16"  : metrics_16,
        "bfloat16" : metrics_b16
    }
})

with open("metrics.json", "w") as fd:
    json_dump(metrics, fd)

print("Parameters and metrics saved")
print(BARLENGTH * BARCHAR)



# plotting results
os.mkdir("images")
os.chdir("images")
os.mkdir("dirty")
print("Plotting results...")



# image_data_analysis.create_histogram(dirty_image_exponents_64, "histogram-dirty_image-exponents.png")
# image_data_analysis.create_histogram(dirty_image_mantissas_64, "histogram-dirty_image-mantissas.png")

# image_data_analysis.create_histogram(psf_exponents_64, "histogram-psf-exponents.png")
# image_data_analysis.create_histogram(psf_mantissas_64, "histogram-psf-mantissas.png")

# image_data_analysis.create_histogram(vis_real_exponents_64, "histogram-visibilities-real_part-exponents.png")
# image_data_analysis.create_histogram(vis_real_mantissas_64, "histogram-visibilities-real_part-mantissas.png")

# image_data_analysis.create_histogram(vis_imag_exponents_64, "histogram-visibilities-imaginary_part-exponents.png")
# image_data_analysis.create_histogram(vis_imag_mantissas_64, "histogram-visibilities-imaginary_part-mantissas.png")



fig, axes = plt.subplots(figsize=(14, 12))


# dirty image
for p in [90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
    percentile = np.percentile(dirty_image_64, p)
    dirty_image_new = np.clip(dirty_image_64, -np.nanstd(dirty_image_64), percentile)

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


# used memory consumption comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["memory_consumption_megabytes"]["float64"]["total"]["used"],
        metrics["memory_consumption_megabytes"]["float32"]["total"]["used"],
        metrics["memory_consumption_megabytes"]["float16"]["total"]["used"],
        metrics["memory_consumption_megabytes"]["bfloat16"]["total"]["used"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("Memory consumption by data type")
plt.ylabel("Memory consumed (MB)")
plt.savefig(f"memory-consumption.png", format="png")


# peak memory consumption comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["memory_consumption_megabytes"]["float64"]["total"]["peak"],
        metrics["memory_consumption_megabytes"]["float32"]["total"]["peak"],
        metrics["memory_consumption_megabytes"]["float16"]["total"]["peak"],
        metrics["memory_consumption_megabytes"]["bfloat16"]["total"]["peak"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("Peak memory consumption by data type")
plt.ylabel("Memory consumed (MB)")
plt.savefig(f"peak-memory-consumption.png", format="png")


# time consumption comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["computation_time_seconds"]["float64"]["total"],
        metrics["computation_time_seconds"]["float16"]["total"],
        metrics["computation_time_seconds"]["float32"]["total"],
        metrics["computation_time_seconds"]["bfloat16"]["total"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("Computing time by data type")
plt.ylabel("Time lapsed (s)")
plt.savefig(f"computing-time.png", format="png")


# RMS comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["image_metrics"]["float64"]["rms"],
        metrics["image_metrics"]["float16"]["rms"],
        metrics["image_metrics"]["float32"]["rms"],
        metrics["image_metrics"]["bfloat16"]["rms"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("RMS value of the clean model by data type")
plt.ylabel("RMS value")
plt.savefig(f"rms.png", format="png")


# dynamic range comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["image_metrics"]["float64"]["dynamic_range"],
        metrics["image_metrics"]["float16"]["dynamic_range"],
        metrics["image_metrics"]["float32"]["dynamic_range"],
        metrics["image_metrics"]["bfloat16"]["dynamic_range"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("Dynamic range of the clean model by data type")
plt.ylabel("Dynamic range (dB)")
plt.savefig(f"dynamic-range.png", format="png")


# SNR comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["image_metrics"]["float64"]["snr"],
        metrics["image_metrics"]["float16"]["snr"],
        metrics["image_metrics"]["float32"]["snr"],
        metrics["image_metrics"]["bfloat16"]["snr"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("SNR of the clean model by data type")
plt.ylabel("SNR (dB)")
plt.savefig(f"snr.png", format="png")


# PSNR comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["image_metrics"]["float64"]["psnr"],
        metrics["image_metrics"]["float16"]["psnr"],
        metrics["image_metrics"]["float32"]["psnr"],
        metrics["image_metrics"]["bfloat16"]["psnr"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("PSNR of the clean model by data type")
plt.ylabel("PSNR (dB)")
plt.savefig(f"psnr.png", format="png")


# SSIM comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["image_metrics"]["float64"]["ssim"],
        metrics["image_metrics"]["float16"]["ssim"],
        metrics["image_metrics"]["float32"]["ssim"],
        metrics["image_metrics"]["bfloat16"]["ssim"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("SSIM of the clean model by data type")
plt.ylabel("SSIM")
plt.savefig(f"ssim.png", format="png")


print("Results plotted and saved to PNG files")
