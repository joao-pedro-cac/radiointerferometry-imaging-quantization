#!/usr/bin/env python3
"""
Collect metrics.json (and parameters.json) from thousands of experiment
folders into a single pandas DataFrame, saved as CSV.

Usage:
    python analyse_metrics.py /path/to/root_folder -o results.csv

Expected layout (folder names may vary slightly, that's OK):
    root_folder/
        gaussian_dr100_s0__q-all-bf16__vf025/
            results-08-Aug-2026/
                experiment_14-16-17/
                    metrics.json
                    parameters.json   (optional)
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Pattern used to pull a few extra columns out of the experiment folder name,
# e.g. "gaussian_dr100_s0__q-all-bf16__vf025"
FOLDER_PATTERN = re.compile(
    r"^(?P<sky_model>[A-Za-z0-9]+)_dr(?P<dr>\d+)_s(?P<s>\d+)"
    r"__q-(?P<quant>[A-Za-z0-9\-]+)__vf(?P<vf>\d+)$"
)


def flatten(data: dict, prefix: str = "") -> dict:
    """Turn a nested dict into a flat one, e.g. {"a": {"b": 1}} -> {"a.b": 1}."""
    flat = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, new_key))
        else:
            flat[new_key] = value
    return flat


def parse_folder_name(name: str) -> dict:
    """Extract dr / s / q / vf from the experiment folder name, if it matches."""
    match = FOLDER_PATTERN.match(name)
    if not match:
        return {}
    fields = match.groupdict()
    return {
        "sky_model": fields["sky_model"],
        "dynamic_range": int(fields["dr"]),
        "seed": int(fields["s"]),
        "quantization_type": fields["quant"],
        "visibilities_fraction": int(fields["vf"]) / 100,
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_row(metrics_path: Path, root: Path) -> dict:
    """Build one CSV row from a single metrics.json file."""
    experiment_folder = metrics_path.relative_to(root).parts[0]

    row = {
        "experiment_folder": experiment_folder,
        # "metrics_path": str(metrics_path),
    }
    row.update(parse_folder_name(experiment_folder))
    row.update(flatten(load_json(metrics_path), prefix="metrics"))

    params_path = metrics_path.parent / "parameters.json"
    if params_path.exists():
        row.update(flatten(load_json(params_path), prefix="param"))

    return row


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files into a CSV.")
    parser.add_argument("root", type=Path, help="Root folder containing the experiment folders")
    parser.add_argument("-o", "--output", type=Path, default=Path("metrics_summary.csv"))
    args = parser.parse_args()

    metrics_files = sorted(args.root.rglob("metrics.json"))
    print(f"Found {len(metrics_files)} metrics.json file(s) under {args.root}")

    rows = []
    for metrics_path in metrics_files:
        try:
            rows.append(build_row(metrics_path, args.root))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skipped] {metrics_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {args.output}")



    # data plotting
    stages = ["visibilities", "dirty_image", "psf", "clean_model"]
    quant_types = ["float64", "float32", "float16", "bfloat16"]
    image_metrics = ["cross_correlation", "psnr", "achieved_dr", "residual_rms", "residual_rms_over_sigma", "flux_ratio"]
    x_col = "dynamic_range"


    try:
        os.mkdir("data-comparison")
    except:
        pass
    os.chdir("data-comparison")


    for metric in image_metrics:
        y_col = f"metrics.image_metrics.{metric}"

        for stage in stages:
            stage_col = f"param.quantization.{stage}"

            fig, ax = plt.subplots(figsize=(7, 5))
            for q_type in quant_types:
                subset = df[df[stage_col] == q_type]
                averaged = subset.groupby(x_col)[y_col].median().reset_index()
                ax.plot(averaged[x_col], averaged[y_col], label=q_type, marker='.')

            ax.set_title(f"{stage} quantization")
            ax.set_xlabel("base_dynamic_range")
            ax.set_ylabel(f"{metric}")
            ax.legend(title="Quantization type")
            fig.tight_layout()
            plt.savefig(f"{metric}-{stage}")
            plt.close()
    
    
    # memory consumption
    for metric in ["dirty_image", "psf", "clean_image", "total"]:
        y_col = f"metrics.memory_consumption_megabytes.{metric}"

        for stage in stages:
            stage_col = f"param.quantization.{stage}"

            fig, ax = plt.subplots(figsize=(7, 5))
            for q_type in quant_types:
                subset = df[df[stage_col] == q_type]
                averaged = subset.groupby(x_col)[y_col].mean().reset_index()
                ax.plot(averaged[x_col], averaged[y_col], label=q_type, marker='.')

            ax.set_title(f"{stage} quantization")
            ax.set_xlabel("base_dynamic_range")
            ax.set_ylabel(f"memory_consumption_{metric} (MB)")
            ax.legend(title="Quantization type")
            fig.tight_layout()
            plt.savefig(f"memory_consumption-{metric}-{stage}")
            plt.close()


    # computation time
    for metric in ["dirty_image", "psf", "clean_image", "total"]:
        y_col = f"metrics.computation_time_seconds.{metric}"

        for stage in stages:
            stage_col = f"param.quantization.{stage}"

            fig, ax = plt.subplots(figsize=(7, 5))
            for q_type in quant_types:
                subset = df[df[stage_col] == q_type]
                averaged = subset.groupby(x_col)[y_col].mean().reset_index()
                ax.plot(averaged[x_col], averaged[y_col], label=q_type, marker='.')

            ax.set_title(f"{stage} quantization")
            ax.set_xlabel("base_dynamic_range")
            ax.set_ylabel(f"computation_time_{metric} (s)")
            ax.legend(title="Quantization type")
            fig.tight_layout()
            plt.savefig(f"computation_time-{metric}-{stage}")
            plt.close()


    print(f"Data saved in {Path("./").absolute()}/data-comparison")



if __name__ == "__main__":
    main()







# import os
# import matplotlib.pyplot as plt
# from sys import argv
# from json import load as json_load

# if len(argv) < 2:
#     raise Exception("Error, insert a text file")

# analysis_filepath = argv[1]
# with open(analysis_filepath, "rt") as fd:
#     info = json_load(fd)

#     output_name = info["name"]
#     output_directory = info["output_directory"]
#     labels = list(info["input_directories"].keys())

#     computation_time_dirty = []
#     computation_time_psf = []
#     computation_time_clean = []
#     computation_time_total = []

#     memory_consumption_dirty = []
#     memory_consumption_psf = []
#     memory_consumption_clean = []
#     memory_consumption_total = []

#     rms = []
#     dr = []
#     snr = []
#     psnr = []
#     ssim = []

#     for directory in labels:
#         with open(info["input_directories"][directory] + "/metrics.json", "rt") as metrics_file:
#             metrics = json_load(metrics_file)

#             computation_time_dirty.append(metrics["computation_time_seconds"]["dirty_image"])
#             computation_time_psf.append(metrics["computation_time_seconds"]["psf"])
#             computation_time_clean.append(metrics["computation_time_seconds"]["clean_image"])
#             computation_time_total.append(metrics["computation_time_seconds"]["total"])

#             memory_consumption_dirty.append(metrics["memory_consumption_megabytes"]["dirty_image"])
#             memory_consumption_psf.append(metrics["memory_consumption_megabytes"]["psf"])
#             memory_consumption_clean.append(metrics["memory_consumption_megabytes"]["clean_image"])
#             memory_consumption_total.append(metrics["memory_consumption_megabytes"]["total"])
            
#             rms.append(metrics["image_metrics"]["rms"])
#             dr.append(metrics["image_metrics"]["dynamic_range"])
#             snr.append(metrics["image_metrics"]["snr"])
#             psnr.append(metrics["image_metrics"]["psnr"])
#             ssim.append(metrics["image_metrics"]["ssim"])

#     os.chdir(info["output_directory"])
#     os.mkdir(info["name"])
#     os.chdir(info["name"])



#     # RMS
#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, rms)
#     ax.set(ylabel="RMS value", title="RMS value comparison")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("rms.png", format="png")


#     # dynamic range
#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, dr)
#     ax.set(ylabel="Dynamic range (dB)", title="Dynamic range comparison")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("dynamic_range.png", format="png")
    

#     # SNR
#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, snr)
#     ax.set(ylabel="SNR value (dB)", title="SNR value comparison")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("snr.png", format="png")


#     # PSNR
#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, psnr)
#     ax.set(ylabel="PSNR value (dB)", title="PSNR value comparison")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("psnr.png", format="png")


#     # SSIM
#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, ssim)
#     ax.set(ylabel="SSIM value", title="SSIM index comparison")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("ssim.png", format="png")


#     # computation time
#     os.mkdir("computation-time")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, computation_time_dirty)
#     ax.set(ylabel="Time elapsed (s)", title="Dirty image computation time")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("computation-time/computation-time-dirtyimage.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, computation_time_psf)
#     ax.set(ylabel="Time elapsed (s)", title="PSF computation time")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("computation-time/computation-time-psf.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, computation_time_clean)
#     ax.set(ylabel="Time elapsed (s)", title="Clean model computation time")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("computation-time/computation-time-cleanmodel.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, computation_time_total)
#     ax.set(ylabel="Time elapsed (s)", title="Total computation time")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("computation-time/computation-time-total.png", format="png")


#     # memory consumption
#     os.mkdir("memory-consumption")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, memory_consumption_dirty)
#     ax.set(ylabel="Memory growth (MB)", title="Dirty image memory consumption")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("memory-consumption/memory-consumption-dirtyimage.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, memory_consumption_psf)
#     ax.set(ylabel="Memory growth (MB)", title="PSF memory consumption")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("memory-consumption/memory-consumption-psf.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, memory_consumption_clean)
#     ax.set(ylabel="Memory growth (MB)", title="Clean model memory consumption")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("memory-consumption/memory-consumption-cleanmodel.png", format="png")

#     fig, ax = plt.subplots(figsize=(20, 15))
#     bars = plt.bar(labels, memory_consumption_total)
#     ax.set(ylabel="Memory growth (MB)", title="Total memory consumption")
#     plt.bar_label(bars, fmt="{:.6f}")
#     plt.savefig("memory-consumption/memory-consumption-total.png", format="png")
