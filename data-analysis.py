import os
import matplotlib.pyplot as plt
from sys import argv
from json import load as json_load

if len(argv) < 2:
    raise Exception("Error, insert a text file")

analysis_filepath = argv[1]
with open(analysis_filepath, "rt") as fd:
    info = json_load(fd)

    output_name = info["name"]
    output_directory = info["output_directory"]
    labels = list(info["input_directories"].keys())

    computation_time_dirty = []
    computation_time_psf = []
    computation_time_clean = []
    computation_time_total = []

    memory_consumption_dirty = []
    memory_consumption_psf = []
    memory_consumption_clean = []
    memory_consumption_total = []

    rms = []
    dr = []
    snr = []
    psnr = []
    ssim = []

    for directory in labels:
        with open(info["input_directories"][directory] + "/metrics.json", "rt") as metrics_file:
            metrics = json_load(metrics_file)

            computation_time_dirty.append(metrics["computation_time_seconds"]["dirty_image"])
            computation_time_psf.append(metrics["computation_time_seconds"]["psf"])
            computation_time_clean.append(metrics["computation_time_seconds"]["clean_image"])
            computation_time_total.append(metrics["computation_time_seconds"]["total"])

            memory_consumption_dirty.append(metrics["memory_consumption_megabytes"]["dirty_image"])
            memory_consumption_psf.append(metrics["memory_consumption_megabytes"]["psf"])
            memory_consumption_clean.append(metrics["memory_consumption_megabytes"]["clean_image"])
            memory_consumption_total.append(metrics["memory_consumption_megabytes"]["total"])
            
            rms.append(metrics["image_metrics"]["rms"])
            dr.append(metrics["image_metrics"]["dynamic_range"])
            snr.append(metrics["image_metrics"]["snr"])
            psnr.append(metrics["image_metrics"]["psnr"])
            ssim.append(metrics["image_metrics"]["ssim"])

    os.chdir(info["output_directory"])
    os.mkdir(info["name"])
    os.chdir(info["name"])



    # RMS
    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, rms)
    ax.set(ylabel="RMS value", title="RMS value comparison")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("rms.png", format="png")


    # dynamic range
    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, dr)
    ax.set(ylabel="Dynamic range (dB)", title="Dynamic range comparison")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("dynamic_range.png", format="png")
    

    # SNR
    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, snr)
    ax.set(ylabel="SNR value (dB)", title="SNR value comparison")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("snr.png", format="png")


    # PSNR
    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, psnr)
    ax.set(ylabel="PSNR value (dB)", title="PSNR value comparison")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("psnr.png", format="png")


    # SSIM
    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, ssim)
    ax.set(ylabel="SSIM value", title="SSIM index comparison")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("ssim.png", format="png")


    # computation time
    os.mkdir("computation-time")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, computation_time_dirty)
    ax.set(ylabel="Time elapsed (s)", title="Dirty image computation time")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("computation-time/computation-time-dirtyimage.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, computation_time_psf)
    ax.set(ylabel="Time elapsed (s)", title="PSF computation time")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("computation-time/computation-time-psf.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, computation_time_clean)
    ax.set(ylabel="Time elapsed (s)", title="Clean model computation time")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("computation-time/computation-time-cleanmodel.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, computation_time_total)
    ax.set(ylabel="Time elapsed (s)", title="Total computation time")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("computation-time/computation-time-total.png", format="png")


    # memory consumption
    os.mkdir("memory-consumption")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, memory_consumption_dirty)
    ax.set(ylabel="Memory growth (MB)", title="Dirty image memory consumption")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("memory-consumption/memory-consumption-dirtyimage.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, memory_consumption_psf)
    ax.set(ylabel="Memory growth (MB)", title="PSF memory consumption")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("memory-consumption/memory-consumption-psf.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, memory_consumption_clean)
    ax.set(ylabel="Memory growth (MB)", title="Clean model memory consumption")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("memory-consumption/memory-consumption-cleanmodel.png", format="png")

    fig, ax = plt.subplots(figsize=(20, 15))
    bars = plt.bar(labels, memory_consumption_total)
    ax.set(ylabel="Memory growth (MB)", title="Total memory consumption")
    plt.bar_label(bars, fmt="{:.6f}")
    plt.savefig("memory-consumption/memory-consumption-total.png", format="png")
