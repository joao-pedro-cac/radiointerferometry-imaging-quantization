#!/usr/bin/env python3
"""Generate an OSKAR sky model (point or Gaussian) for the quantization study.

One invocation -> one .osm (+ truth catalog). Sweep DR and seed externally.
Intra-field DR = S_max/S_min is pinned exactly to --dr (brightest and faintest
sources forced), so realized DR == nominal DR — no under-sampling of the faint
end, which the old uniform(1, DR) draw suffered from.
"""
import argparse
import csv
import os
import sys

import numpy as np

# OSKAR fixed-format column order:
#   RA Dec I Q U V freq0 spix RM maj min pa
#   deg deg Jy - - - Hz  -    -  arcsec arcsec deg     (points: maj=min=pa=0)


def sample_fluxes(rng, n, peak_flux, dr, dist):
    s_max, s_min = peak_flux, peak_flux / dr
    interior = n - 2
    if dist == "loguniform":
        mid = 10.0 ** rng.uniform(np.log10(s_min), np.log10(s_max), interior)
    else:  # linear: under-samples faint end, kept only for comparison
        mid = rng.uniform(s_min, s_max, interior)
    flux = np.concatenate(([s_max, s_min], mid)) if interior > 0 else np.array([s_max, s_min])
    rng.shuffle(flux)  # decorrelate flux rank from source index
    return flux


def sample_positions(rng, n, ra0, dec0, fov_deg):
    # angularly-square field: scale RA offset by 1/cos(dec0)
    cosd = np.cos(np.radians(dec0))
    dra = rng.uniform(-fov_deg / 2, fov_deg / 2, n) / cosd
    ddec = rng.uniform(-fov_deg / 2, fov_deg / 2, n)
    radius = np.hypot(dra * cosd, ddec)  # angular offset from phase centre [deg]
    return ra0 + dra, dec0 + ddec, radius


def make_sources(args, rng):
    # flux + position drawn first and identically for both kinds, so a point
    # field and a Gaussian field at the same seed share positions and fluxes.
    flux = sample_fluxes(rng, args.num_sources, args.peak_flux, args.dr, args.flux_dist)
    ra, dec, radius = sample_positions(rng, args.num_sources, args.ra0, args.dec0, args.fov_deg)
    n = args.num_sources
    if args.kind == "point":
        maj = minr = pa = np.zeros(n)
    else:
        maj = rng.uniform(args.maj_min, args.maj_max, n)
        minr = maj * rng.uniform(args.axis_ratio_min, 1.0, n)
        pa = rng.uniform(0.0, 180.0, n)
    return dict(ra=ra, dec=dec, radius=radius, flux=flux, maj=maj, min=minr, pa=pa)


def write_osm(path, s, args):
    g = lambda v: f"{v:.10g}"
    hdr = [
        f"# OSKAR sky model | type={args.kind} n={args.num_sources} dr={args.dr:g} "
        f"seed={args.seed} peak={args.peak_flux:g}Jy fov={args.fov_deg:g}deg\n",
        f"# realized_intrafield_dr={s['flux'].max()/s['flux'].min():.6g} "
        f"s_max={s['flux'].max():.6g} s_min={s['flux'].min():.6g}\n",
        "# RA Dec I Q U V freq0 spix RM maj min pa\n",
    ]
    body = [
        f"{g(s['ra'][i])} {g(s['dec'][i])} {g(s['flux'][i])} 0 0 0 "
        f"{g(args.freq0)} {g(args.spix)} 0 {g(s['maj'][i])} {g(s['min'][i])} {g(s['pa'][i])}\n"
        for i in range(args.num_sources)
    ]
    with open(path, "wt") as fd:
        fd.writelines(hdr + body)


# def write_truth(path, s, args):
#     with open(path, "wt", newline="") as fd:
#         w = csv.writer(fd)
#         w.writerow(["name", "type", "ra_deg", "dec_deg", "radius_deg",
#                     "flux_jy", "maj_arcsec", "min_arcsec", "pa_deg"])
#         for i in range(args.num_sources):
#             w.writerow([f"s{i}", args.kind, f"{s['ra'][i]:.10g}", f"{s['dec'][i]:.10g}",
#                         f"{s['radius'][i]:.6g}", f"{s['flux'][i]:.6g}",
#                         f"{s['maj'][i]:.6g}", f"{s['min'][i]:.6g}", f"{s['pa'][i]:.6g}"])


def build_parser():
    p = argparse.ArgumentParser(description="OSKAR sky-model generator for the quantization study.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", help="output .osm path (auto-named if omitted)")
    common.add_argument("-n", "--num-sources", type=int, default=50)
    common.add_argument("--ra0", type=float, default=0.0, help="phase-centre RA [deg]")
    common.add_argument("--dec0", type=float, default=-30.0, help="phase-centre Dec [deg]")
    common.add_argument("--fov-deg", type=float, default=1.0, help="angular field width [deg]")
    common.add_argument("--dr", type=float, default=1000.0,
                        help="intra-field DR S_max/S_min (pinned exactly)")
    common.add_argument("--peak-flux", type=float, default=1.0,
                        help="S_max [Jy]; with sigma sets thermal DR = peak/sigma")
    common.add_argument("--flux-dist", choices=["loguniform", "uniform"], default="loguniform")
    common.add_argument("--freq0", type=float, default=1.4e9, help="reference frequency [Hz]")
    common.add_argument("--spix", type=float, default=-0.7, help="spectral index")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--sigma", type=float, default=None,
                        help="thermal noise [Jy]; if set, warns when faintest-source SNR < --min-snr")
    common.add_argument("--min-snr", type=float, default=5.0)

    sub = p.add_subparsers(dest="kind", required=True)
    sub.add_parser("point", parents=[common], help="point sources")
    gp = sub.add_parser("gaussian", parents=[common], help="Gaussian components")
    gp.add_argument("--maj-min", type=float, default=5.0, help="min major FWHM [arcsec]")
    gp.add_argument("--maj-max", type=float, default=60.0, help="max major FWHM [arcsec]")
    gp.add_argument("--axis-ratio-min", type=float, default=0.3, help="min minor/major ratio")
    return p


def validate(args):
    if args.num_sources < 2:
        sys.exit("num-sources must be >= 2 to pin the intra-field DR")
    if args.dr < 1:
        sys.exit("dr must be >= 1")
    if args.kind == "gaussian" and not (0 < args.axis_ratio_min <= 1):
        sys.exit("axis-ratio-min must be in (0, 1]")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    out = args.out or f"{args.kind}_n{args.num_sources}_dr{args.dr:g}_seed{args.seed}.osm"
    rng = np.random.default_rng(args.seed)
    s = make_sources(args, rng)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_osm(out, s, args)
    print(f"wrote {out}  (realized intra-field DR = {s['flux'].max()/s['flux'].min():.6g})")

    if args.sigma is not None:  # guard: faint source must sit above thermal, else it's noise-limited not precision-limited
        snr = s["flux"].min() / args.sigma
        print(f"faintest-source SNR = {snr:.3g} (thermal DR = {args.peak_flux/args.sigma:.3g})")
        if snr < args.min_snr:
            print(f"WARNING: faintest source at {snr:.2g}sigma < {args.min_snr}sigma — "
                  "raise --peak-flux or lower --dr", file=sys.stderr)

    # if not args.no_truth:
        # truth = os.path.splitext(out)[0] + ".truth.csv"
        # write_truth(truth, s, args)
        # print(f"wrote {truth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())