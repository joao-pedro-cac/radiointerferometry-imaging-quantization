import os
import threading
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import quantization.ieee_754_casting as ieee_casting
from misc.memmon import *
from misc.fileread import *
from astropy.io import fits
from calendar import month_abbr
from time import time, localtime
from json import dump as json_dump
from json import load as json_load
from imaging.image_pipeline import ImagePipeline



# experiment time and day
year, month, day, hour, min, sec, _, _, _ = localtime()
day_dirname = f"results-{day}-{month_abbr[month]}-{year}"
time_dirname = f"experiment_{hour}-{min}-{sec}"


with open("simulation-config.json", "r") as fd:
    simulation_config = json_load(fd)


# auxiliary variables
BARLENGTH = 80
BARCHAR = '-'
# VISIBILITY_MAXNUMBER = 4096
# SIMULATED_DATA_PATH = "../data/simulated/point_extended/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"
# TRUE_MODEL_PATH = "../data/simulated/point_extended/truth_model.fits"
SIMULATED_DATA_PATH = simulation_config["file_paths"]["astronomical_data_zarr_file_path"]
TRUE_MODEL_PATH = simulation_config["file_paths"]["true_model_file_path"]



# image reconstruction parameters
GRIDDING_EPSILON      = simulation_config["computation_parameters"]["gridding_epsilon"]
CLEAN_VARIANT         = simulation_config["computation_parameters"]["clean_variant"]
CLEAN_GAMMA           = simulation_config["computation_parameters"]["clean_gamma"]
CLEAN_PF              = simulation_config["computation_parameters"]["clean_peak_fraction"]
CLEAN_MAXITER         = simulation_config["computation_parameters"]["clean_max_iterations"]
MAJORLOOP_MAXITER     = simulation_config["computation_parameters"]["feedback_loop_max_iterations"]
SAFE_FLOAT16_MAX      = simulation_config["computation_parameters"]["safe_float16_max"]
experiment_commentary = simulation_config["computation_parameters"]["experiment_commentary"]
enable_log            = simulation_config["computation_parameters"]["enable_log"]



assert SAFE_FLOAT16_MAX < 65504
# CLEAN_VARIANT = "clark"
# GRIDDING_EPSILON = 1e-5
# CLEAN_GAMMA = 0.0125
# CLEAN_PF = 0.0075
# CLEAN_MAXITER = 5000
# MAJORLOOP_MAXITER = 20
# SAFE_FLOAT16_MAX = 10000.0
# experiment_commentary = "Comparing float64 x float32 x float16 x bfloat16."



########################################################################################################################
########################################################################################################################
########################################################################################################################



# data set simulated by OSKAR
telescope = simulation_config["attributes"]["telescope"]
skymodel = simulation_config["attributes"]["sky_model"]
true_image = np.squeeze(fits.getdata(TRUE_MODEL_PATH))       # true (original) sky image
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_PATH, engine="zarr")



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

pipeline.set_true_model(TRUE_MODEL_PATH)




########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float64 image pipeline ################################################

# visibilities type casting
vis_64 = vis_data.astype("complex128")
wgt_64 = weight_data.astype("float64")




#  dirty image generation
print("Generating float64 dirty image...")
dirty_image_computing_time_64 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
dirty_image_64 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_64,
                                              wgt=wgt_64,
                                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image_64 = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computing_time_64 = time() - dirty_image_computing_time_64
print("float64 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_64} s")
print(f"RAM consumption: {mem_dirty_image_64} MB")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float64 PSF...")
psf_computing_time_64 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
psf_64 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_64,
                              wgt=wgt_64,
                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_psf_64 = monitor.get_consumed_ram() / (1024 ** 2)
psf_computing_time_64 = time() - psf_computing_time_64
print("float64 PSF generated")
print(f"Computing time: {psf_computing_time_64} s")
print(f"RAM consumption: {mem_psf_64} MB")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float64 clean model...")
clean_image_computing_time_64 = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model_64, status_64 = pipeline.compute_clean_image(dirty_image_64, psf_64)


# pipeline loop
k = 1
while k <= MAJORLOOP_MAXITER and status_64 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_64,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_64)
    residual_vis = vis_64 - recomputed_vis
    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt_64)
    clean_components, status_64 = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                               psf=psf_64)
    clean_model_64 += clean_components
    k += 1

monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image_64 = monitor.get_consumed_ram() / (1024 ** 2)

clean_image_computing_time_64 = time() - clean_image_computing_time_64
print("float64 clean model generated")
print(f"Computing time: {clean_image_computing_time_64} s")
print(f"RAM consumption: {mem_clean_image_64} MB")
print(BARCHAR * BARLENGTH)



########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float32 image pipeline ################################################

# visibilities type casting
vis_32 = vis_data.astype("complex64")
wgt_32 = weight_data.astype("float32")




#  dirty image generation
print("Generating float32 dirty image...")
dirty_image_computing_time_32 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
dirty_image_32 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_32,
                                              wgt=wgt_32,
                                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image_32 = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computing_time_32 = time() - dirty_image_computing_time_32
print("float64 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_32} s")
print(f"RAM consumption: {mem_dirty_image_32} MB")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float32 PSF...")
psf_computing_time_32 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
psf_32 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_32,
                              wgt=wgt_32,
                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_psf_32 = monitor.get_consumed_ram() / (1024 ** 2)
psf_computing_time_32 = time() - psf_computing_time_32
print("float32 PSF generated")
print(f"Computing time: {psf_computing_time_32} s")
print(f"RAM consumption: {mem_psf_32} MB")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float32 clean model...")
clean_image_computing_time_32 = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model_32, status_32 = pipeline.compute_clean_image(dirty_image_32, psf_32)


# pipeline loop
k = 1
while k <= MAJORLOOP_MAXITER and status_32 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_32,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_32)
    residual_vis = vis_32 - recomputed_vis
    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt_32)
    
    clean_components, status_32 = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                               psf=psf_32)
    clean_model_32 += clean_components
    k += 1

monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image_32 = monitor.get_consumed_ram() / (1024 ** 2)
clean_image_computing_time_32 = time() - clean_image_computing_time_32
print("float32 clean model generated")
print(f"Computing time: {clean_image_computing_time_32} s")
print(f"RAM consumtion: {mem_clean_image_32} MB")
print(BARCHAR * BARLENGTH)




########################################################################################################################
########################################################################################################################
########################################################################################################################

################################################ float16 image pipeline ################################################

# visibilities type casting
scale_factor_vis = SAFE_FLOAT16_MAX / np.max(np.abs(vis_data))   # float16 scalar to maintain the visibilities values in the proper dynamic range
scale_factor_wgt = 1.0 / np.max(np.abs(weight_data))             # weight normalization

vis_16 = (vis_32 * scale_factor_vis).astype(np.complex64)
wgt_16 = (wgt_32 * scale_factor_wgt).astype(np.float32)

for i in range(vis_16.shape[0]):
    for j in range(vis_16.shape[1]):
        data = complex(vis_16[i, j])
        vis_16[i, j] = ieee_casting.float_to_half(data.real)[0] + ieee_casting.float_to_half(data.imag)[0] * 1j
for i in range(wgt_16.shape[0]):
    for j in range(wgt_16.shape[1]):
        wgt_16[i, j] = ieee_casting.float_to_half(wgt_16[i, j])[0]




#  dirty image generation
print("Generating float16 dirty image...")
dirty_image_computing_time_16 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
dirty_image_16 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_16,
                                              wgt=wgt_16,
                                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image_16 = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computing_time_16 = time() - dirty_image_computing_time_16
print("float16 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_16} s")
print(f"RAM consumption: {mem_dirty_image_16} MB")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating float16 PSF...")
psf_computing_time_16 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
psf_16 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_16,
                              wgt=wgt_16,
                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_psf_16 = monitor.get_consumed_ram() / (1024 ** 2)
psf_computing_time_16 = time() - psf_computing_time_16
print("float16 PSF generated")
print(f"Computing time: {psf_computing_time_16} s")
print(f"RAM consumption: {mem_psf_16} MB")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating float16 clean model...")
clean_image_computing_time_16 = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model_16, status_16 = pipeline.compute_clean_image(dirty_image_16, psf_16)


# pipeline loop
k = 1
while k <= MAJORLOOP_MAXITER and status_16 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_16,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_16)
    residual_vis = vis_16 - recomputed_vis
    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt_16)
    
    clean_components, status_16 = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                               psf=psf_16)
    clean_model_16 += clean_components
    k += 1
clean_model_16 = clean_model_16 / scale_factor_vis

monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image_16 = monitor.get_consumed_ram() / (1024 ** 2)

clean_image_computing_time_16 = time() - clean_image_computing_time_16
print("float16 clean model generated")
print(f"Computing time: {clean_image_computing_time_16} s")
print(f"RAM consumption: {mem_clean_image_16} MB")
print(BARCHAR * BARLENGTH)




########################################################################################################################
########################################################################################################################
########################################################################################################################

############################################# brain float16 image pipeline #############################################

# visibilities type casting
vis_b16 = vis_32.copy()
for i in range(vis_b16.shape[0]):
    for j in range(vis_b16.shape[1]):
        data = complex(vis_b16[i, j])
        vis_b16[i, j] = ieee_casting.float_to_bfloat(data.real)[0] + ieee_casting.float_to_bfloat(data.imag)[0] * 1j
wgt_b16 = wgt_32.copy()
for i in range(wgt_b16.shape[0]):
    for j in range(wgt_b16.shape[1]):
        wgt_b16[i, j] = ieee_casting.float_to_bfloat(wgt_b16[i, j])[0]




#  dirty image generation
print("Generating bfloat16 dirty image...")
dirty_image_computing_time_b16 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
dirty_image_b16 = pipeline.compute_dirty_image(uvw=uvw_data,
                                              freq=freq_data,
                                              vis=vis_b16,
                                              wgt=wgt_b16,
                                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image_b16 = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computing_time_b16 = time() - dirty_image_computing_time_b16
print("bfloat16 dirty image generated")
print(f"Computing time: {dirty_image_computing_time_b16} s")
print(f"RAM consumption: {mem_dirty_image_b16} MB")
print(BARCHAR * BARLENGTH)




# PSF generation
print("Generating bfloat16 PSF...")
psf_computing_time_b16 = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()
psf_b16 = pipeline.compute_psf(uvw=uvw_data,
                              freq=freq_data,
                              vis=vis_b16,
                              wgt=wgt_b16,
                              verbosity=False)
monitor.keep_measuring = False
monitor_thread.join()
mem_psf_b16 = monitor.get_consumed_ram() / (1024 ** 2)
psf_computing_time_b16 = time() - psf_computing_time_b16
print("bfloat16 PSF generated")
print(f"Computing time: {psf_computing_time_b16} s")
print(f"RAM consumption: {mem_psf_b16} MB")
print(BARCHAR * BARLENGTH)




# CLEAN algorithm
print("Generating bfloat16 clean model...")
clean_image_computing_time_b16 = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model_b16, status_b16 = pipeline.compute_clean_image(dirty_image_b16, psf_b16)


# pipeline loop
k = 1
while k <= MAJORLOOP_MAXITER and status_b16 == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model_b16,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt_b16)
    residual_vis = vis_b16 - recomputed_vis
    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt_b16)
    
    clean_components, status_b16 = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                               psf=psf_b16)
    clean_model_b16 += clean_components
    k += 1

monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image_b16 = monitor.get_consumed_ram() / (1024 ** 2)

clean_image_computing_time_b16 = time() - clean_image_computing_time_b16
print("bfloat16 clean model generated")
print(f"Computing time: {clean_image_computing_time_b16} s")
print(f"RAM consumption: {mem_clean_image_b16} MB")
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

clean_model_64  = np.transpose(clean_model_64)[::-1, :]
clean_model_32  = np.transpose(clean_model_32)[::-1, :]
clean_model_16  = np.transpose(clean_model_16)[::-1, :]
clean_model_b16 = np.transpose(clean_model_b16)[::-1, :]


# saving clean image to a FITS file
try:
    # copy the original model's coordinate system (header)
    with fits.open(TRUE_MODEL_PATH) as hdul:
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
except FileNotFoundError:      # exception if TRUE_MODEL_PATH is not found
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
    "telescope"           : telescope,
    "sky_model"           : skymodel,
    "data set"            : {
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
    "daytime"             : {
        "day"    : day,
        "month"  : month_abbr[month],
        "year"   : year,
        "hour"   : hour,
        "minute" : min,
        "second" : sec
    },
    "clean_algorithm"     : CLEAN_VARIANT,
    "loop_iterations"     : MAJORLOOP_MAXITER,
    "dirty_image_epsilon" : GRIDDING_EPSILON,
    "clean_image"         : {
        "gamma"          : CLEAN_GAMMA,
        "peak_fraction"  : CLEAN_PF,
        "max_iterations" : CLEAN_MAXITER
    },
    "commentary"          : experiment_commentary
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
            "dirty_image" : mem_dirty_image_64,
            "psf"         : mem_psf_64,
            "clean_image" : mem_clean_image_64,
            "total"       : mem_dirty_image_64 + mem_psf_64 + mem_clean_image_64
        },
        "float32"  : {
            "dirty_image" : mem_dirty_image_32,
            "psf"         : mem_psf_32,
            "clean_image" : mem_clean_image_32,
            "total"       : mem_dirty_image_32 + mem_psf_32 + mem_clean_image_32
        },
        "float16"  : {
            "dirty_image" : mem_dirty_image_16,
            "psf"         : mem_psf_16,
            "clean_image" : mem_clean_image_16,
            "total"       : mem_dirty_image_16 + mem_psf_16 + mem_clean_image_16
        },
        "bfloat16" : {
            "dirty_image" : mem_dirty_image_b16,
            "psf"         : mem_psf_b16,
            "clean_image" : mem_clean_image_b16,
            "total"       : mem_dirty_image_b16 + mem_psf_b16 + mem_clean_image_b16
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


# memory consumption comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["memory_consumption_megabytes"]["float64"]["total"],
        metrics["memory_consumption_megabytes"]["float32"]["total"],
        metrics["memory_consumption_megabytes"]["float16"]["total"],
        metrics["memory_consumption_megabytes"]["bfloat16"]["total"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("Memory consumption by data type")
plt.ylabel("Memory consumed (MB)")
plt.savefig(f"memory-consumption.png", format="png")


# time consumption comparison
plt.figure()
labels = ["float64", "float32", "float16", "bfloat16"]
nums = [metrics["computation_time_seconds"]["float64"]["total"],
        metrics["computation_time_seconds"]["float32"]["total"],
        metrics["computation_time_seconds"]["float16"]["total"],
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
        metrics["image_metrics"]["float32"]["rms"],
        metrics["image_metrics"]["float16"]["rms"],
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
        metrics["image_metrics"]["float32"]["dynamic_range"],
        metrics["image_metrics"]["float16"]["dynamic_range"],
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
        metrics["image_metrics"]["float32"]["snr"],
        metrics["image_metrics"]["float16"]["snr"],
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
        metrics["image_metrics"]["float32"]["psnr"],
        metrics["image_metrics"]["float16"]["psnr"],
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
        metrics["image_metrics"]["float32"]["ssim"],
        metrics["image_metrics"]["float16"]["ssim"],
        metrics["image_metrics"]["bfloat16"]["ssim"]]
bars = plt.bar(labels, nums)
plt.bar_label(bars, fmt='{:.6f}')
plt.title("SSIM of the clean model by data type")
plt.ylabel("SSIM")
plt.savefig(f"ssim.png", format="png")


print("Results plotted and saved to PNG files")
