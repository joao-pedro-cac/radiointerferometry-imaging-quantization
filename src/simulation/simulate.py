"""
Simulate a radio interferometric observation and inject heteroscedastic noise.

Pipeline
--------
1. Generate OSKAR settings.ini from CLI args (no hardcoded paths).
2. Run OSKAR binary → obs.ms with DATA = pure sky model response, no system noise.
3. Read noiseless DATA → write truth_model.fits and truth_dirty.fits.
4. Inject heteroscedastic Gaussian noise into DATA column (noise.py).
5. Optionally run pfb init → obs_I.xds (--run-init flag).

All outputs go to {output-dir}/{scenario-name}/:
    obs.ms/
    obs_sigma.zarr/     ground truth σ_i per visibility + grid parameters as attrs
    settings.ini        OSKAR settings used (reproducibility record)
    truth_model.fits    rendered sky model image (delta functions at source positions)
    truth_dirty.fits    dirty image of noiseless visibilities (PSF × sky model)
    obs_I.xds/          pfb init output (only if --run-init)

Usage
-----
source scripts/env.sh
python experiments/simulate.py \\
    --scenario-name point_field_eps05_k10 \\
    --telescope meerkat.tm \\
    --skymodel point_field.osm \\
    --ra-deg -120.0 --dec-deg -60.0 \\
    --start-freq-hz 1.4e9 --nchan 10 --freq-inc-hz 1e6 \\
    --obs-length-s 25200 --ntime-steps 2520 \\
    --sigma0 1e-4 --outlier-fraction 0.05 --outlier-scale 10 --seed 42 \\
    --fov-deg 1.0 \\
    --oskar-bin $OSKAR_BIN \\
    --output-dir $DATA_ROOT/data/simulated \\
    --data-root $DATA_ROOT \\
    --nthreads 8
"""

import sys
import shutil
import argparse
import subprocess
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import zarr
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from ducc0.wgridder.experimental import vis2dirty
from pfb_imaging.core.init import init as pfb_init
from pfb_imaging.utils.fits import save_fits, set_wcs
from pfb_imaging.utils.misc import set_image_size

from noise import add_heteroscedastic_noise


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario-name", required=True)
    p.add_argument("--telescope", required=True, help="Filename under data/oskar/telescopes/")
    p.add_argument("--skymodel", required=True, help="Filename under data/oskar/skymodels/")
    p.add_argument("--ra-deg", type=float, required=True, help="Phase centre RA in degrees")
    p.add_argument("--dec-deg", type=float, required=True, help="Phase centre Dec in degrees")
    p.add_argument("--start-freq-hz", type=float, required=True)
    p.add_argument("--nchan", type=int, required=True)
    p.add_argument("--freq-inc-hz", type=float, required=True)
    p.add_argument("--obs-length-s", type=float, required=True, help="Total observation length in seconds")
    p.add_argument("--ntime-steps", type=int, required=True)
    p.add_argument("--sigma0", type=float, required=True, help="Baseline noise std in Jy")
    p.add_argument("--outlier-fraction", type=float, default=0.0, help="Fraction of visibilities with inflated noise")
    p.add_argument("--outlier-scale", type=float, default=1.0, help="σ_outlier = outlier-scale * sigma0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oskar", default="/usr/local/bin/oskar_sim_interferometer",
                   help="Path to oskar_sim_interferometer binary, or a Singularity/Apptainer "
                        "image (.sif/.simg). Overridden by the OSKAR env variable.")
    p.add_argument("--output-dir", required=True, help="Root dir; scenario subdir created inside")
    p.add_argument("--data-root", required=True, help="Root of the research repo")
    p.add_argument("--fov-deg", type=float, required=True, help="Image field of view in degrees (sets nx for truth images and pfb grid)")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--sr-factor", type=float, default=2.0,
                   help="Super-resolution factor: cell = λ_min / (2 * uv_max * sr_factor). "
                        "Lower values → coarser grid → faster Clark minor loop.")
    p.add_argument("--run-init", action="store_true", help="Also run pfb init to produce obs_I.xds")
    p.add_argument("--skip-oskar", action="store_true",
                   help="Skip OSKAR step; assume obs.ms already exists (useful when OSKAR ran natively)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scenario directory")
    return p


def generate_settings_ini(args, scenario_dir: Path, ms_path: Path) -> Path:
    """Write OSKAR settings.ini with absolute paths. Returns path to the file."""
    data_root = Path(args.data_root)
    telescope_path = data_root / "data" / "oskar" / "telescopes" / args.telescope
    skymodel_path = data_root / "data" / "oskar" / "skymodels" / args.skymodel
    time_avg = args.obs_length_s / args.ntime_steps

    content = textwrap.dedent(f"""\
        [simulator]
        use_gpus=false

        [observation]
        start_frequency_hz={args.start_freq_hz}
        num_channels={args.nchan}
        frequency_inc_hz={args.freq_inc_hz}
        start_time_utc=2000-01-01T00:00:00
        length={args.obs_length_s}
        num_time_steps={args.ntime_steps}
        phase_centre_ra_deg={args.ra_deg}
        phase_centre_dec_deg={args.dec_deg}

        [telescope]
        input_directory={telescope_path}

        [sky]
        oskar_sky_model/file={skymodel_path}

        [interferometer]
        ms_filename={ms_path}
        channel_bandwidth_hz={args.freq_inc_hz}
        time_average_sec={time_avg:.6f}
    """)

    settings_path = scenario_dir / "settings.ini"
    settings_path.write_text(content)
    return settings_path


def build_oskar_cmd(oskar: str, settings_path: Path, bind_paths: list) -> list:
    """Build the OSKAR subprocess command for a native binary or Singularity image."""
    if oskar.endswith(".sif") or oskar.endswith(".simg"):
        binds = ",".join(str(Path(p).resolve()) for p in bind_paths)
        return ["singularity", "exec", "--bind", binds, oskar,
                "oskar_sim_interferometer", str(settings_path)]
    return [oskar, str(settings_path)]


def run_oskar(oskar: str, settings_path: Path, bind_paths: list) -> None:
    """Run OSKAR interferometer simulator as a subprocess."""
    cmd = build_oskar_cmd(oskar, settings_path, bind_paths)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OSKAR failed (exit {result.returncode}):\n{result.stderr}")


def parse_osm(osm_path: Path) -> list[tuple]:
    """
    Parse OSKAR sky model file.

    Returns list of (ra_deg, dec_deg, flux_Jy, spectral_index, freq0_Hz).
    Lines starting with '#' and empty lines are skipped.
    """
    sources = []
    with open(osm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ra, dec, flux = float(parts[0]), float(parts[1]), float(parts[2])
            freq0, spix = float(parts[6]), float(parts[7])
            sources.append((ra, dec, flux, spix, freq0))
    return sources


def write_truth_images(
    ms_path: Path,
    osm_path: Path,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    nthreads: int,
    output_dir: Path,
    sr_factor: float = 2.0,
) -> dict:
    """
    Generate truth_model.fits and truth_dirty.fits from noiseless MS.

    truth_dirty : dirty image of the noiseless model visibilities; shows how
                  the PSF interacts with the sky model.
    truth_model : rendered point sources at their sky positions; a sparse image
                  with one pixel per source, used to verify source placement.

    Must be called before noise injection (while DATA is still noiseless).

    Returns
    -------
    dict with keys: fov_deg, sr_factor, nx, ny, cell_deg
        Grid parameters persisted into obs_sigma.zarr attrs by main().
    """
    # --- Read MS ---
    datasets = list(xds_from_ms(str(ms_path), columns=["DATA", "UVW"]))
    uvw = np.concatenate([ds.UVW.data.compute() for ds in datasets], axis=0).astype(np.float64)
    vis_full = np.concatenate([ds.DATA.data.compute() for ds in datasets], axis=0)
    # (nrow_total, nchan, ncorr) → Stokes I = mean of parallel hands
    vis_I = 0.5 * (vis_full[:, :, 0] + vis_full[:, :, -1])  # (nrow, nchan), cast to complex128 for float64 gridding
    vis_I = vis_I.astype(np.complex128)

    spw_ds = xds_from_table(f"{ms_path}::SPECTRAL_WINDOW", columns=["CHAN_FREQ"])[0]
    freq = spw_ds.CHAN_FREQ.data.compute().ravel().astype(np.float64)  # (nchan,)

    # --- Compute image parameters ---
    # sr_factor controls resolution: lower → coarser grid → smaller image → faster Clark
    uv_max = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max()
    max_freq = freq.max()
    nx, ny, _, _, _, cell_rad, cell_deg = set_image_size(
        uv_max, max_freq, fov_deg, sr_factor, None, None, None, 1.4
    )
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    ref_freq = float(np.mean(freq))
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, [ra_rad, dec_rad], ref_freq)

    # --- truth_dirty: vis2dirty of noiseless Stokes I visibilities ---
    wgt = np.ones((vis_I.shape[0], vis_I.shape[1]), dtype=np.float64)
    dirty = np.zeros((nx, ny), dtype=np.float64)
    vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_I,
        wgt=wgt,
        npix_x=nx,
        npix_y=ny,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        center_x=0.0,
        center_y=0.0,
        epsilon=1e-5,
        flip_u=False,
        flip_v=True,
        flip_w=False,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=nthreads,
        sigma_min=1.1,
        sigma_max=3.0,
        dirty=dirty,
    )
    dirty /= wgt.sum()
    save_fits(dirty, str(output_dir / "truth_dirty.fits"), hdr)

    # --- truth_model: delta functions at source pixel positions ---
    model = np.zeros((nx, ny), dtype=np.float64)
    sources = parse_osm(osm_path)
    for ra_s, dec_s, flux, spix, freq0 in sources:
        # l, m direction cosines (small-angle exact for sources near phase centre)
        dra = np.deg2rad(ra_s) - ra_rad
        l = np.cos(dec_rad) * np.sin(dra)  # noqa: E741
        m = np.sin(np.deg2rad(dec_s)) * np.cos(dec_rad) - np.cos(np.deg2rad(dec_s)) * np.sin(dec_rad) * np.cos(dra)
        flux_mfs = flux * (ref_freq / freq0) ** spix
        px = int(round(nx // 2 - l / cell_rad))
        py = int(round(ny // 2 + m / cell_rad))
        if 0 <= px < nx and 0 <= py < ny:
            model[px, py] += flux_mfs
    save_fits(model, str(output_dir / "truth_model.fits"), hdr)

    return {"fov_deg": fov_deg, "sr_factor": sr_factor, "nx": nx, "ny": ny, "cell_deg": float(cell_deg)}


def run_pfb_init(ms_path: Path, scenario_dir: Path, nthreads: int) -> None:
    """Run pfb init to produce obs_I.xds in scenario_dir."""
    pfb_init(
        ms=[ms_path],
        output_filename=str(scenario_dir / "obs"),
        nthreads=nthreads,
        overwrite=False,
    )


def main() -> None:
    args = get_parser().parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    scenario_dir = output_dir / args.scenario_name

    # Guard against accidental overwrite.
    # When --skip-oskar is set, preserve obs.ms and settings.ini; only derived
    # files (zarr stores, FITS truth images) are rewritten by the steps below.
    if scenario_dir.exists() and not args.overwrite:
        print(f"Scenario directory already exists: {scenario_dir}")
        print("Use --overwrite to replace it.")
        sys.exit(1)
    if scenario_dir.exists() and args.overwrite and not args.skip_oskar:
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    ms_path = scenario_dir / "obs.ms"
    osm_path = data_root / "data" / "oskar" / "skymodels" / args.skymodel

    # 1. Generate settings.ini + run OSKAR (skip if --skip-oskar, e.g. when OSKAR ran natively)
    if not args.skip_oskar:
        import os
        oskar = os.environ.get("OSKAR", args.oskar)
        settings_path = generate_settings_ini(args, scenario_dir, ms_path)
        print(f"[simulate] Written {settings_path}")
        print(f"[simulate] Running OSKAR ({oskar}) ...")
        run_oskar(oskar, settings_path, [data_root, output_dir])

        print(f"[simulate] OSKAR done → {ms_path}")
    else:
        if not ms_path.exists():
            raise FileNotFoundError(f"--skip-oskar set but obs.ms not found at {ms_path}")
        print(f"[simulate] Skipping OSKAR (obs.ms already present at {ms_path})")

    # 2. Truth images — must run before noise injection while DATA is still noiseless
    print(f"[simulate] Writing truth images (noiseless DATA)...")
    grid_meta = write_truth_images(ms_path, osm_path, args.ra_deg, args.dec_deg, args.fov_deg, args.nthreads, scenario_dir, sr_factor=args.sr_factor)
    print(f"[simulate] Written truth_model.fits and truth_dirty.fits")

    # 3. Inject heteroscedastic noise
    print(f"[simulate] Injecting noise (epsilon={args.outlier_fraction}, scale={args.outlier_scale})...")
    sigma = add_heteroscedastic_noise(
        ms_path=ms_path,
        sigma0=args.sigma0,
        epsilon=args.outlier_fraction,
        scale=args.outlier_scale,
        seed=args.seed,
    )
    print(f"[simulate] Noise injected. σ range: [{sigma.min():.3e}, {sigma.max():.3e}] Jy")


    # Write grid parameters into obs_sigma.zarr attrs so imaging scripts inherit the same grid
    sigma_zarr_path = scenario_dir / "obs_sigma.zarr"
    zarr.open_group(str(sigma_zarr_path), mode="a").attrs.update(grid_meta)

    # 5. Optional pfb init
    if args.run_init:
        print(f"[simulate] Running pfb init...")
        run_pfb_init(ms_path, scenario_dir, args.nthreads)
        print(f"[simulate] pfb init done → {scenario_dir / 'obs_I.xds'}")

    print(f"[simulate] Done. Scenario at {scenario_dir}")


if __name__ == "__main__":
    main()
