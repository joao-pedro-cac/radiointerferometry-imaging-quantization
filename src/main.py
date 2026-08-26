import os
import threading
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import quantization.ieee_754_casting as ieee_casting
from sys import argv
from math import ceil
from misc.memmon import *
from misc.fileread import *
from astropy.io import fits
from calendar import month_abbr
from time import time, localtime
from json import dump as json_dump
from json import load as json_load
from quantization.quantize import *
from imaging.image_data_analysis import *
from imaging.image_pipeline import ImagePipeline

from pfb_imaging.utils.naming import xds_from_url
from daskms.fsspec_store import DaskMSStore

from pathlib import Path


# experiment time and day
year, month, day, hour, min, sec, _, _, _ = localtime()
day_dirname = f"results-{day:02}-{month_abbr[month]}-{year}"
time_dirname = f"experiment_{hour:02}-{min:02}-{sec:02}"




# make sure there is at least one command-line argument
if len(argv) < 2:
    raise Exception("Error, insert an input JSON file")

# verify if the argument file is a JSON
simulation_config_filepath = argv[1]
if ".json" not in simulation_config_filepath:
    raise Exception("Invalid file, the program input must be a JSON file")

with open(simulation_config_filepath, "r") as fd:
    simulation_config = json_load(fd)




# auxiliary variables
BARLENGTH = 80
BARCHAR = '-'
DATA_PATH = simulation_config["file_paths"]["data_path"]
DATASET_NAME = simulation_config["file_paths"]["dataset_name"]
SIMULATED_DATA_PATH = Path(DATA_PATH, DATASET_NAME, "obs_I.xds")
TRUE_MODEL_PATH = Path(DATA_PATH, DATASET_NAME, "truth_model.fits")
PERCENTAGE_VISIBILITIES = simulation_config["attributes"]["vis_fraction"]
RNG_SEED = simulation_config["attributes"]["rng_seed"]

assert PERCENTAGE_VISIBILITIES >= 0 and PERCENTAGE_VISIBILITIES <= 1
assert type(RNG_SEED) == int and RNG_SEED >= 0




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
noise_floor = simulation_config["attributes"]["noise_floor"]

# oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_PATH, engine="zarr")

## Open dataset from xds radio dataset using pfb-imaging utils functions.

xds_store = DaskMSStore(SIMULATED_DATA_PATH)

print("Opening xds dataset.... ")
print("Warning : Assuming one xarray in xds. If using multiple scan datasets only first one will be processed.")

xds, _ = xds_from_url(xds_store.url)
oskar_simulated_dataset = xds[0]



rng = np.random.default_rng(RNG_SEED)


# visibilities data subset


uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]


# nvis_tot = len(uvw)
# nvis = rate_vis * nvis_tot
# idx =  rng.permutation(nvis_tot)[0:nvis]# random permutation d'indice de taille nvis

uvw_data    = uvw.data[:ceil(PERCENTAGE_VISIBILITIES * uvw.shape[0])]
freq_data   = freq.data
vis_data    = np.squeeze(vis.data)[:ceil(PERCENTAGE_VISIBILITIES * vis.shape[1])]
weight_data = np.squeeze(weight.data)[:ceil(PERCENTAGE_VISIBILITIES * weight.shape[1])]



########################################################################################################################
########################################################################################################################
########################################################################################################################



# image processing pipeline object
pipeline = ImagePipeline(clean_algorithm=CLEAN_VARIANT,
                         gridding_epsilon=GRIDDING_EPSILON,
                         clean_gamma=CLEAN_GAMMA,
                         clean_pf=CLEAN_PF,
                         clean_maxiter=CLEAN_MAXITER)

pipeline.set_true_model(TRUE_MODEL_PATH)
pipeline.set_noise_floor(noise_floor)


########################################################################################################################
########################################################################################################################
########################################################################################################################



# quantization parameters
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

# visibilities and weights quantization
if enable_log:
    print(f"Quantizing visibilities and weights to {vis_quantization_type}...")

vis, scalefactor_vis = quantize_visibilities(vis_data, vis_quantization_type, SAFE_FLOAT16_MAX)
wgt, _ = quantize_weights(weight_data, vis_quantization_type, SAFE_FLOAT16_MAX)

if enable_log:
    print(BARCHAR * BARLENGTH)




#  dirty image
if enable_log:
    print(f"Generating {dirty_image_quantization_type} dirty image...")

dirty_image_computation_time = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                           freq=freq_data,
                                           vis=vis,
                                           wgt=wgt,
                                           verbosity=False)
dirty_image, scalefactor_dirty = quantize_image(dirty_image, dirty_image_quantization_type, SAFE_FLOAT16_MAX)


monitor.keep_measuring = False
monitor_thread.join()
mem_dirty_image = monitor.get_consumed_ram() / (1024 ** 2)
dirty_image_computation_time = time() - dirty_image_computation_time

if enable_log:
    print(f"{dirty_image_quantization_type} dirty image generated")
    print(f"Computation time: {dirty_image_computation_time} s")
    print(f"RAM consumption: {mem_dirty_image} MB")
    print(BARCHAR * BARLENGTH)




# PSF
if enable_log:
    print(f"Generating {psf_quantization_type} PSF...")

psf_computation_time = time()
monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


psf = pipeline.compute_psf(uvw=uvw_data,
                           freq=freq_data,
                           vis=vis,
                           wgt=wgt,
                           verbosity=False)
pipeline.set_psf(psf)
psf, scalefactor_psf = quantize_image(psf, psf_quantization_type, SAFE_FLOAT16_MAX)


monitor.keep_measuring = False
monitor_thread.join()
mem_psf = monitor.get_consumed_ram() / (1024 ** 2)
psf_computation_time = time() - psf_computation_time

if enable_log:
    print("float16 PSF generated")
    print(f"Computation time: {psf_computation_time} s")
    print(f"RAM consumption: {mem_psf} MB")
    print(BARCHAR * BARLENGTH)




# CLEAN algorithm
if enable_log:
    print(f"Generating {clean_model_quantization_type} clean model...")

clean_image_computation_time = time()

monitor = MemoryMonitor()
monitor_thread = threading.Thread(target=monitor.measure_usage)
monitor_thread.start()


clean_model, status = pipeline.compute_clean_image(dirty_image, psf, verbosity=enable_log)
clean_model, scalefactor_clean = quantize_image(clean_model, clean_model_quantization_type, SAFE_FLOAT16_MAX)

# pipeline loop
k = 1
prev_peak = 0

while k <= MAJORLOOP_MAXITER:
    recomputed_vis = pipeline.compute_visibilities(dirty_image=clean_model,
                                                   uvw=uvw_data,
                                                   freq=freq_data,
                                                   wgt=wgt)
    recomputed_vis, _ = quantize_visibilities(recomputed_vis, vis_quantization_type, SAFE_FLOAT16_MAX, True, scalefactor_vis)
    residual_vis = vis - recomputed_vis

    residual_dirty_image = pipeline.compute_dirty_image(uvw=uvw_data,
                                                        freq=freq_data,
                                                        vis=residual_vis,
                                                        wgt=wgt)
    residual_dirty_image, scalefactor_dirty = quantize_image(residual_dirty_image, dirty_image_quantization_type, SAFE_FLOAT16_MAX)
    


    clean_components, status = pipeline.compute_clean_image(dirty_image=residual_dirty_image,
                                                            psf=psf,
                                                            verbosity=enable_log)
    clean_model += clean_components
    clean_model, scaled_factor_clean = quantize_image(clean_model, clean_model_quantization_type, SAFE_FLOAT16_MAX)

    res_mad = 1.4826*np.median(np.abs(residual_dirty_image - np.median(residual_dirty_image)))
    res_peak = np.max(np.abs(residual_dirty_image))
    prev_residual = residual_dirty_image.copy()   # keep last good one

     
    if res_peak == 0: 
        residual_dirty_image = prev_residual       # metrics use the finite residual
        break


    residual_dirty_image = residual_dirty_image / res_peak
    if res_peak < 3*res_mad or (prev_peak - res_peak) < 0.02*prev_peak:
        break
    prev_peak = res_peak
    k += 1


# clean model de-scaling for every scaled variable due to float16 limited value range
if vis_quantization_type == "float16":
    clean_model = (clean_model / scalefactor_vis).astype(np.float32)
    residual_dirty_image = (residual_dirty_image / scalefactor_vis).astype(np.float32)
if dirty_image_quantization_type == "float16":
    clean_model = (clean_model / scalefactor_dirty).astype(np.float32)
    residual_dirty_image = (residual_dirty_image / scalefactor_dirty).astype(np.float32)
if psf_quantization_type == "float16":
    clean_model = (clean_model * scalefactor_psf).astype(np.float32)
    residual_dirty_image = (residual_dirty_image / scalefactor_psf).astype(np.float32)
if clean_model_quantization_type == "float16":
    clean_model = (clean_model / scalefactor_clean).astype(np.float32)
    residual_dirty_image = (residual_dirty_image / scaled_factor_clean).astype(np.float32)


monitor.keep_measuring = False
monitor_thread.join()
mem_clean_image = monitor.get_consumed_ram() / (1024 ** 2)

clean_image_computation_time = time() - clean_image_computation_time

if enable_log:
    print(f"{clean_model_quantization_type} clean model generated")
    print(f"Computation time: {clean_image_computation_time} s")
    print(f"RAM consumption: {mem_clean_image} MB")
    print(BARCHAR * BARLENGTH)



########################################################################################################################
########################################################################################################################
########################################################################################################################



# save results and metrics
output_filepath = simulation_config["file_paths"]["output_file_path"]
os.makedirs(output_filepath, exist_ok=True)
os.chdir(output_filepath)
print(f"Output in {output_filepath}")

try:
    os.mkdir(day_dirname)

    if enable_log:
        print("New day of experiments: new directory created")
except:
    pass

if enable_log:
    print(f"Experiment day: {day:02} {month_abbr[month]} {year}")

os.chdir(day_dirname)
os.mkdir(time_dirname)
os.chdir(time_dirname)

if enable_log:
    print(f"Experiment began at {hour:02}:{min:02}:{sec:02}")
    print(BARLENGTH * BARCHAR)
    # print("Saving CLEAN image to FITS files...")


# # --- restored image: sky-oriented model+psf in, display frame written out ---
# save_restored_fits(clean_model, psf, "restored_model.fits",
#                    template_header_path=TRUE_MODEL_PATH)


dirty_image  = np.transpose(dirty_image)[::-1, :]
clean_model  = np.transpose(clean_model)[::-1, :]


# # saving clean image to a FITS file
# try:
#     # copy the original model's coordinate system (header)
#     with fits.open(TRUE_MODEL_PATH) as hdul:
#         original_header = hdul[0].header
        
#     # create a new FITS files using the clean image and the copied header
#     new_hdu = fits.PrimaryHDU(data=clean_model, header=original_header)
#     new_hdu.writeto("clean_model.fits", overwrite=True)

#     if enable_log:
#         print(f"Successfully saved CLEAN image")
# except FileNotFoundError:      # exception if TRUE_MODEL_PATH is not found
#     new_hdu = fits.PrimaryHDU(data=clean_model)
#     new_hdu.writeto("clean_model.fits", overwrite=True)

#     if enable_log:
#         print(f"Saved CLEAN images (without header metadata)")

if enable_log:
    print(BARLENGTH * BARCHAR)
    print("Saving experiment parameters and result metrics...")



parameters = {
    "telescope"           : telescope,
    "sky_model"           : skymodel,
    "percentage_visibilities" : PERCENTAGE_VISIBILITIES,
    "rng_seed"                : RNG_SEED,
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
            "dirty_image" : dirty_image_computation_time,
            "psf"         : psf_computation_time,
            "clean_image" : clean_image_computation_time,
            "total"       : dirty_image_computation_time + psf_computation_time + clean_image_computation_time
    },
    "memory_consumption_megabytes" : {
            "dirty_image" : mem_dirty_image,
            "psf"         : mem_psf,
            "clean_image" : mem_clean_image,
            "total"       : mem_dirty_image + mem_psf + mem_clean_image
    }
}

# metrics.update({ "image_metrics" : pipeline.calculate_image_metrics(clean_model) })

# after the major loop, before metrics — reads only variables already in scope
if not np.isfinite(res_peak) or res_peak == 0:
    converged_reason = "zero_residual"
elif res_peak < 3*res_mad:
    converged_reason = "threshold"
elif k >= MAJORLOOP_MAXITER:
    converged_reason = "max_major"
else:
    converged_reason = "stall"


metrics.update({"image_metrics": pipeline.calculate_image_metrics(clean_model, residual_dirty_image)})
metrics["image_metrics"]["converged_reason"] = converged_reason
metrics["image_metrics"]["n_major_cycles"] = int(k)

with open("metrics.json", "w") as fd:
    json_dump(metrics, fd)

if enable_log:
    print(BARLENGTH * BARCHAR)



########################################################################################################################
########################################################################################################################
########################################################################################################################



# plotting results
os.mkdir("images")
os.chdir("images")
# os.mkdir("dirty")

# if enable_log:
#     print("Plotting results...")


# fig, axes = plt.subplots(figsize=(14, 12))


# # dirty image
# for p in [90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
#     percentile = np.percentile(dirty_image, p)
#     dirty_image_new = np.clip(dirty_image, -np.nanstd(dirty_image), percentile)

#     im = axes.imshow(dirty_image_new, cmap='inferno', origin='lower')
#     axes.set_title(f'Dirty Image (percentile = {p}%)')
#     plt.tight_layout()

#     if int(p) == p:
#         filename = str(p)
#     else:
#         whole_part = int(p)
#         decimal_part = p - whole_part
#         filename = str(whole_part) + '_' + str(decimal_part)[2:5]
#     plt.savefig(f'dirty/dirty-image-{filename}.png', format='png')



# bit length histograms
os.mkdir("bitlength")
os.chdir("bitlength")

os.mkdir("visibilities")
os.mkdir("visibilities/real")
os.mkdir("visibilities/imaginary")
os.mkdir("dirty-image")
os.mkdir("psf")
os.mkdir("clean-model")



# visibilities (real part)
exp = []
man = []

for i in range(vis.shape[0]):
    for j in range(vis.shape[1]):
        if vis_quantization_type == "float64":
            _, e, m = ieee_casting.get_double_datafields(vis[i, j].real)
            exp.append(e)
            man.append(m)
        elif vis_quantization_type == "float32":
            _, e, m = ieee_casting.get_float_datafields(vis[i, j].real)
            exp.append(e)
            man.append(m)
        elif vis_quantization_type == "float16":
            _, e, m = ieee_casting.get_half_datafields(vis[i, j].real)
            exp.append(e)
            man.append(m)
        else:
            _, e, m = ieee_casting.get_bfloat_datafields(vis[i, j].real)
            exp.append(e)
            man.append(m)

create_bitlength_histogram(exp, "visibilities/real/exponent.png", "Effective bit length of the exponent field (visibilities, real part)")
create_bitlength_histogram(man, "visibilities/real/mantissa.png", "Effective bit length of the mantissa field (visibilities, real part)")



# visibilities (imaginary part)
exp = []
man = []

for i in range(vis.shape[0]):
    for j in range(vis.shape[1]):
        if vis_quantization_type == "float64":
            _, e, m = ieee_casting.get_double_datafields(vis[i, j].imag)
            exp.append(e)
            man.append(m)
        elif vis_quantization_type == "float32":
            _, e, m = ieee_casting.get_float_datafields(vis[i, j].imag)
            exp.append(e)
            man.append(m)
        elif vis_quantization_type == "float16":
            _, e, m = ieee_casting.get_half_datafields(vis[i, j].imag)
            exp.append(e)
            man.append(m)
        else:
            _, e, m = ieee_casting.get_bfloat_datafields(vis[i, j].imag)
            exp.append(e)
            man.append(m)

create_bitlength_histogram(exp, "visibilities/imaginary/exponent.png", "Effective bit length of the exponent field (visibilities, imaginary part)")
create_bitlength_histogram(man, "visibilities/imaginary/mantissa.png", "Effective bit length of the mantissa field (visibilities, imaginary part)")



# dirty image
exp = []
man = []

for i in range(dirty_image.shape[0]):
    for j in range(dirty_image.shape[1]):
        if dirty_image_quantization_type == "float64":
            _, e, m = ieee_casting.get_double_datafields(dirty_image[i, j])
            exp.append(e)
            man.append(m)
        elif dirty_image_quantization_type == "float32":
            _, e, m = ieee_casting.get_float_datafields(dirty_image[i, j])
            exp.append(e)
            man.append(m)
        elif dirty_image_quantization_type == "float16":
            _, e, m = ieee_casting.get_half_datafields(dirty_image[i, j])
            exp.append(e)
            man.append(m)
        else:
            _, e, m = ieee_casting.get_bfloat_datafields(dirty_image[i, j])
            exp.append(e)
            man.append(m)

create_bitlength_histogram(exp, "dirty-image/exponent.png", "Effective bit length of the exponent field (dirty image)")
create_bitlength_histogram(man, "dirty-image/mantissa.png", "Effective bit length of the mantissa field (dirty image)")



# PSF
exp = []
man = []

for i in range(psf.shape[0]):
    for j in range(psf.shape[1]):
        if psf_quantization_type == "float64":
            _, e, m = ieee_casting.get_double_datafields(psf[i, j])
            exp.append(e)
            man.append(m)
        elif psf_quantization_type == "float32":
            _, e, m = ieee_casting.get_float_datafields(psf[i, j])
            exp.append(e)
            man.append(m)
        elif psf_quantization_type == "float16":
            _, e, m = ieee_casting.get_half_datafields(psf[i, j])
            exp.append(e)
            man.append(m)
        else:
            _, e, m = ieee_casting.get_bfloat_datafields(psf[i, j])
            exp.append(e)
            man.append(m)

create_bitlength_histogram(exp, "psf/exponent.png", "Effective bit length of the exponent field (psf)")
create_bitlength_histogram(man, "psf/mantissa.png", "Effective bit length of the mantissa field (psf)")



# clean model
exp = []
man = []

for i in range(clean_model.shape[0]):
    for j in range(clean_model.shape[1]):
        if clean_model_quantization_type == "float64":
            _, e, m = ieee_casting.get_double_datafields(clean_model[i, j])
            exp.append(e)
            man.append(m)
        elif clean_model_quantization_type == "float32":
            _, e, m = ieee_casting.get_float_datafields(clean_model[i, j])
            exp.append(e)
            man.append(m)
        elif clean_model_quantization_type == "float16":
            _, e, m = ieee_casting.get_half_datafields(clean_model[i, j])
            exp.append(e)
            man.append(m)
        else:
            _, e, m = ieee_casting.get_bfloat_datafields(clean_model[i, j])
            exp.append(e)
            man.append(m)

create_bitlength_histogram(exp, "clean-model/exponent.png", "Effective bit length of the exponent field (clean model)")
create_bitlength_histogram(man, "clean-model/mantissa.png", "Effective bit length of the mantissa field (clean model)")


if enable_log:
    print("Results succesfully plotted")