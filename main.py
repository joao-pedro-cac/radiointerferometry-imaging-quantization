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

print("BEFORE:")
print(f"MAX visilities = {np.max(vis_data)}")
print(f"MIN visilities = {np.min(vis_data)}")
print()
print(f"MAX weights = {np.max(weight_data)}")
print(f"MIN weights = {np.min(weight_data)}")
print(f"SUM weights = {np.sum(weight_data)}")
print()
print(f"MAX weighted vis = {np.max(vis_data * weight_data)}")
print(f"MIN weighted vis = {np.min(vis_data * weight_data)}")
print(f"SUM weighted vis = {np.sum(vis_data * weight_data)}")
print()

vis_quantization_type         = simulation_config["quantization"]["visibilities"]
dirty_image_quantization_type = simulation_config["quantization"]["dirty_image"]
psf_quantization_type         = simulation_config["quantization"]["psf"]
clean_model_quantization_type = simulation_config["quantization"]["clean_model"]

assert vis_quantization_type         in ["float64", "float32", "float16", "bfloat16"]
assert dirty_image_quantization_type in ["float64", "float32", "float16", "bfloat16"]
assert psf_quantization_type         in ["float64", "float32", "float16", "bfloat16"]
assert clean_model_quantization_type in ["float64", "float32", "float16", "bfloat16"]



########################################################################################################################
########################################################################################################################
########################################################################################################################

#################################################### image papeline ####################################################

# visibilities type casting
if enable_log:
    print(f"Quantizing visibilities and weights to {vis_quantization_type}...")

if vis_quantization_type == "float64":
    vis = vis_data.astype("complex128")
    wgt = weight_data.astype("float64")
else:
    vis = vis_data.astype("complex64")
    wgt = weight_data.astype("float32")

    # scale_factor_vis = (SAFE_FLOAT16_MAX / np.max(np.abs(vis))) if vis_quantization_type == "float16" else 1.0
    # scale_factor_wgt = (SAFE_FLOAT16_MAX / np.sum(wgt))         if psf_quantization_type == "float16" else (1.0 / np.max(np.abs(wgt)))

    if vis_quantization_type == "float16":
        scale_factor_vis = SAFE_FLOAT16_MAX / np.max(np.abs(vis))
        scale_factor_wgt = SAFE_FLOAT16_MAX / np.sum(wgt)

        vis = (vis * scale_factor_vis).astype(np.complex64)
        wgt = (wgt * scale_factor_wgt).astype(np.float32)

        for i in range(vis.shape[0]):
            for j in range(vis.shape[1]):
                data = complex(vis[i, j])
                vis[i, j] = ieee_casting.float_to_half(data.real)[0] + ieee_casting.float_to_half(data.imag)[0] * 1j

        for i in range(wgt.shape[0]):
            for j in range(wgt.shape[1]):
                wgt[i, j] = ieee_casting.float_to_half(wgt[i, j])[0]
    elif vis_quantization_type == "bfloat16":
        for i in range(vis.shape[0]):
            for j in range(vis.shape[1]):
                data = complex(vis[i, j])
                vis[i, j] = ieee_casting.float_to_bfloat(data.real)[0] + ieee_casting.float_to_bfloat(data.imag)[0] * 1j
        
        for i in range(wgt.shape[0]):
            for j in range(wgt.shape[1]):
                wgt[i, j] = ieee_casting.float_to_bfloat(wgt[i, j])[0]
    else:        # vérifier fonctionnement
        scale_factor_wgt = np.max(wgt) / np.sum(wgt)

if enable_log:
    print("Visibilities and weights quantized")
    print(BARCHAR * BARLENGTH)

print()
print()
print("AFTER:")
print(f"MAX visilities = {np.max(vis)}")
print(f"MIN visilities = {np.min(vis)}")
print()
print(f"MAX weights = {np.max(wgt)}")
print(f"MIN weights = {np.min(wgt)}")
print(f"SUM weights = {np.sum(wgt)}")
print()
print(f"MAX weighted vis = {np.max(vis * wgt)}")
print(f"MIN weighted vis = {np.min(vis * wgt)}")
print(f"SUM weighted vis = {np.sum(vis * wgt)}")
print()
print()




#  dirty image generation
if enable_log:
    print(f"Generating {dirty_image_quantization_type} dirty image...")

dirty_image_computing_time = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()

dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                           freq=freq_data,
                                           vis=vis,
                                           wgt=wgt,
                                           verbosity=False)
print("BEFORE:")
print(f"MAX dirty image = {np.max(dirty_image)}")
print(f"MIN dirty image = {np.min(dirty_image)}")
print(f"SUM dirty image = {np.sum(dirty_image)}")
print()

if dirty_image_quantization_type == "float64":
    dirty_image = dirty_image.astype(np.float64)
else:
    dirty_image = dirty_image.astype(np.float32)

    if dirty_image_quantization_type == "float16":
        scale_factor_dirty = SAFE_FLOAT16_MAX / np.max(np.abs(dirty_image))
        dirty_image = (dirty_image * scale_factor_dirty).astype(np.float32)
        
        for i in range(dirty_image.shape[0]):
            for j in range(dirty_image.shape[1]):
                dirty_image[i, j] = ieee_casting.float_to_half(dirty_image[i, j])[0]

        dirty_image = np.nan_to_num(dirty_image, nan=0.0, posinf=0.0, neginf=0.0)
    elif dirty_image_quantization_type == "bfloat16":
        for i in range(dirty_image.shape[0]):
            for j in range(dirty_image.shape[1]):
                dirty_image[i, j] = ieee_casting.float_to_bfloat(dirty_image[i, j])[0]

monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computing_time = time() - dirty_image_computing_time

if enable_log:
    print(f"{dirty_image_quantization_type} dirty image generated")
    print(f"Computing time: {dirty_image_computing_time} s")
    print(f"RAM consumption: {mem_dirty_image} MB")
    print(BARCHAR * BARLENGTH)
print("AFTER:")
print(f"MAX dirty image = {np.max(dirty_image)}")
print(f"MIN dirty image = {np.min(dirty_image)}")
print(f"SUM dirty image = {np.sum(dirty_image)}")
print()
print()



# PSF generation
if enable_log:
    print(f"Generating {psf_quantization_type} PSF...")

psf_computing_time = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()

psf = pipeline.compute_psf(uvw=uvw_data,
                           freq=freq_data,
                           vis=vis,
                           wgt=wgt,
                           verbosity=False)
print("BEFORE:")
print(f"MAX psf = {np.max(psf)}")
print(f"MIN psf = {np.min(psf)}")
print(f"SUM psf = {np.sum(psf)}")
print()

if psf_quantization_type == "float64":
    psf = psf.astype(np.float64)
else:
    psf = psf.astype(np.float32)

    if psf_quantization_type == "float16":
        scale_factor_psf = SAFE_FLOAT16_MAX / np.max(np.abs(psf))
        psf = (psf * scale_factor_psf).astype(np.float32)

        for i in range(psf.shape[0]):
            for j in range(psf.shape[1]):
                psf[i, j] = ieee_casting.float_to_half(psf[i, j])[0]
        
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)

    elif psf_quantization_type == "bfloat16":
        for i in range(psf.shape[0]):
            for j in range(psf.shape[1]):
                psf[i, j] = ieee_casting.float_to_bfloat(psf[i, j])[0]

monitor.keep_measuring = False
monitor_thread.join()
mem_psf = monitor.get_consumed_ram() / (1024 ** 2)
psf_computing_time = time() - psf_computing_time

if enable_log:
    print("float16 PSF generated")
    print(f"Computing time: {psf_computing_time} s")
    print(f"RAM consumption: {mem_psf} MB")
    print(BARCHAR * BARLENGTH)
print("BEFORE:")
print(f"MAX psf = {np.max(psf)}")
print(f"MIN psf = {np.min(psf)}")
print(f"SUM psf = {np.sum(psf)}")
print()
print()



# CLEAN algorithm
if enable_log:
    print(f"Generating {clean_model_quantization_type} clean model...")

clean_image_computing_time = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model, status = pipeline.compute_clean_image(dirty_image, psf, verbosity=enable_log)


if clean_model_quantization_type == "float64":
    clean_model = clean_model.astype(np.float64)
else:
    clean_model = clean_model.astype(np.float32)

    if clean_model_quantization_type == "float16":
        scale_factor_clean = SAFE_FLOAT16_MAX / np.max(np.abs(clean_model))
        clean_model = (clean_model * scale_factor_clean).astype(np.float32)

        for i in range(clean_model.shape[0]):
            for j in range(clean_model.shape[1]):
                clean_model[i, j] = ieee_casting.float_to_half(clean_model[i, j])[0]
    elif clean_model_quantization_type == "bfloat16":
        for i in range(clean_model.shape[0]):
            for j in range(clean_model.shape[1]):
                clean_model[i, j] = ieee_casting.float_to_bfloat(clean_model[i, j])[0]


# pipeline loop
k = 1
while k <= MAJORLOOP_MAXITER and status == 1:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt)

    if vis_quantization_type == "float64":
        recomputed_vis = recomputed_vis.astype("complex128")
    else:
        recomputed_vis = recomputed_vis.astype("complex64")
        if vis_quantization_type == "float16":
            recomputed_vis = (recomputed_vis * scale_factor_vis).astype(np.complex64)

            for i in range(recomputed_vis.shape[0]):
                for j in range(recomputed_vis.shape[1]):
                    data = complex(recomputed_vis[i, j])
                    recomputed_vis[i, j] = ieee_casting.float_to_half(data.real)[0] + ieee_casting.float_to_half(data.imag)[0] * 1j
        elif vis_quantization_type == "bfloat16":
            for i in range(recomputed_vis.shape[0]):
                for j in range(recomputed_vis.shape[1]):
                    data = complex(recomputed_vis[i, j])
                    recomputed_vis[i, j] = ieee_casting.float_to_bfloat(data.real)[0] + ieee_casting.float_to_bfloat(data.imag)[0] * 1j

    residual_vis = vis - recomputed_vis


    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt)
    
    if dirty_image_quantization_type == "float64":
        residual_dirty_image = residual_dirty_image.astype(np.float64)
    else:
        residual_dirty_image = residual_dirty_image.astype(np.float32)

        if dirty_image_quantization_type == "float16":
            # PEUT-ÊTRE CRÉER UNE NOUVELLE VARIABLE ?
            scale_factor_dirty = SAFE_FLOAT16_MAX / np.max(np.abs(residual_dirty_image))
            residual_dirty_image = (residual_dirty_image * scale_factor_dirty).astype(np.float32)

            for i in range(residual_dirty_image.shape[0]):
                for j in range(residual_dirty_image.shape[1]):
                    residual_dirty_image[i, j] = ieee_casting.float_to_half(residual_dirty_image[i, j])[0]

            residual_dirty_image = np.nan_to_num(residual_dirty_image, nan=0.0, posinf=0.0, neginf=0.0)

        elif dirty_image_quantization_type == "bfloat16":
            for i in range(residual_dirty_image.shape[0]):
                for j in range(residual_dirty_image.shape[1]):
                    residual_dirty_image[i, j] = ieee_casting.float_to_bfloat(residual_dirty_image[i, j])[0]



    clean_components, status = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                            psf=psf,
                                                            verbosity=enable_log)
    clean_model += clean_components

    if clean_model_quantization_type == "float64":
        clean_model = clean_model.astype(np.float64)
    else:
        clean_model = clean_model.astype(np.float32)

        if clean_model_quantization_type == "float16":
            # PEUT-ÊTRE CRÉER UNE NOUVELLE VARIABLE ?
            scale_factor_clean = SAFE_FLOAT16_MAX / np.max(np.abs(clean_model))
            clean_model = (clean_model * scale_factor_clean).astype(np.float32)

            for i in range(clean_model.shape[0]):
                for j in range(clean_model.shape[1]):
                    clean_model[i, j] = ieee_casting.float_to_half(clean_model[i, j])[0]
        elif clean_model_quantization_type == "bfloat16":
            for i in range(clean_model.shape[0]):
                for j in range(clean_model.shape[1]):
                    clean_model[i, j] = ieee_casting.float_to_bfloat(clean_model[i, j])[0]


    k += 1

if vis_quantization_type == "float16":
    clean_model = (clean_model / scale_factor_vis).astype(np.float32)
if dirty_image_quantization_type == "float16":
    clean_model = (clean_model / scale_factor_dirty).astype(np.float32)
if psf_quantization_type == "float16":
    clean_model = (clean_model * scale_factor_psf).astype(np.float32)
if clean_model_quantization_type == "float16":
    clean_model = (clean_model / scale_factor_clean).astype(np.float32)




monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image = monitor.get_consumed_ram() / (1024 ** 2)

clean_image_computing_time = time() - clean_image_computing_time

if enable_log:
    print(f"{clean_model_quantization_type} clean model generated")
    print(f"Computing time: {clean_image_computing_time} s")
    print(f"RAM consumption: {mem_clean_image} MB")
    print(BARCHAR * BARLENGTH)












# save results and metrics
os.chdir("../results")

try:
    os.mkdir(day_dirname)

    if enable_log:
        print("New day of experiments: new directory created")
except:
    pass

if enable_log:
    print(f"Experiment day: {day} {month_abbr[month]} {year}")

os.chdir(day_dirname)
os.mkdir(time_dirname)
os.chdir(time_dirname)

if enable_log:
    print(f"Experiment began at {hour}:{min}:{sec}")
    print(BARLENGTH * BARCHAR)
    print("Saving CLEAN image to FITS files...")


dirty_image  = np.transpose(dirty_image)[::-1, :]
clean_model  = np.transpose(clean_model)[::-1, :]


# saving clean image to a FITS file
try:
    # copy the original model's coordinate system (header)
    with fits.open(TRUE_MODEL_PATH) as hdul:
        original_header = hdul[0].header
        
    # create a new FITS files using the clean image and the copied header
    new_hdu = fits.PrimaryHDU(data=clean_model, header=original_header)
    new_hdu.writeto("clean_model.fits", overwrite=True)

    if enable_log:
        print(f"Successfully saved CLEAN image")
except FileNotFoundError:      # exception if TRUE_MODEL_PATH is not found
    new_hdu = fits.PrimaryHDU(data=clean_model)
    new_hdu.writeto("clean_model.fits", overwrite=True)

    if enable_log:
        print(f"Saved CLEAN images (without header metadata)")

if enable_log:
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
            "type" : vis_quantization_type,
            "size" : vis_data.size
        },
        "weights" : {
            "type" : vis_quantization_type,
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
    "quantization"        : {
        "visibilities" : vis_quantization_type,
        "dirty_image"  : dirty_image_quantization_type,
        "psf"          : psf_quantization_type,
        "clean_model"  : clean_model_quantization_type
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
    "computation_time_seconds" : {
            "dirty_image" : dirty_image_computing_time,
            "psf"         : psf_computing_time,
            "clean_image" : clean_image_computing_time,
            "total"       : dirty_image_computing_time + psf_computing_time + clean_image_computing_time
    },
    "memory_consumption_megabytes" : {
            "dirty_image" : mem_dirty_image,
            "psf"         : mem_psf,
            "clean_image" : mem_clean_image,
            "total"       : mem_dirty_image + mem_psf + mem_clean_image
    }
}

metrics.update({ "image_metrics" : pipeline.calculate_image_metrics(clean_model) })

with open("metrics.json", "w") as fd:
    json_dump(metrics, fd)

if enable_log:
    print("Parameters and metrics saved")
    print(BARLENGTH * BARCHAR)



# plotting results
os.mkdir("images")
os.chdir("images")

if enable_log:
    print("Plotting results...")


fig, axes = plt.subplots(figsize=(14, 12))


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
    plt.savefig(f'dirty-image-{filename}.png', format='png')
