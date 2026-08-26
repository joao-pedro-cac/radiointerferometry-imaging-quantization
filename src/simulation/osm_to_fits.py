import sys
import argparse

import numpy as np

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.4247


def osm_to_fits(osm_path, out_path="sky.fits",
                npix=1024, cell_arcsec=1.0, ra0=None, dec0=None) -> Path:
    """Rasterize an OSKAR sky model (.osm) to a Jy/pixel FITS image for CARTA."""
    rows = []
    for line in Path(osm_path).read_text().splitlines():
        line = line.split("#")[0].strip()            # drop comments/blanks
        if line:
            rows.append(line.replace(",", " ").split())

    ra  = np.array([float(r[0]) for r in rows])
    dec = np.array([float(r[1]) for r in rows])
    ra0  = ra.mean()  if ra0  is None else ra0
    dec0 = dec.mean() if dec0 is None else dec0

    w = WCS(naxis=2)                                  # SIN proj at phase centre
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [npix / 2 + 1, npix / 2 + 1]
    w.wcs.cdelt = [-cell_arcsec / 3600.0, cell_arcsec / 3600.0]
    w.wcs.cunit = ["deg", "deg"]

    img = np.zeros((npix, npix), dtype=np.float32)
    yy, xx = np.mgrid[0:npix, 0:npix]
    x, y = w.all_world2pix(ra, dec, 0)                # 0-indexed pixel coords

    for i, r in enumerate(rows):
        maj = float(r[9]) if len(r) > 9 else 0.0      # arcsec FWHM; 0 => point
        if maj > 0.0:
            img += gaussian_component(r, xx, yy, x[i], y[i], cell_arcsec)
        else:
            xi, yi = round(x[i]), round(y[i])
            if 0 <= xi < npix and 0 <= yi < npix:
                img[yi, xi] += float(r[2])            # Jy in one pixel

    return write_to_fits({"data": img, "wcs": w, "path": out_path})


def gaussian_component(row, xx, yy, xc, yc, cell_arcsec) -> np.ndarray:
    """Elliptical Gaussian on the grid, normalised to integrated flux I (Jy/pixel)."""
    I   = float(row[2])
    sx  = float(row[9])  * FWHM2SIG / cell_arcsec     # sigma in pixels
    sy  = float(row[10]) * FWHM2SIG / cell_arcsec
    pa  = np.radians(float(row[11]))                  # E-of-N; flip sign if mirrored
    dx, dy = xx - xc, yy - yc
    u =  dx * np.cos(pa) + dy * np.sin(pa)
    v = -dx * np.sin(pa) + dy * np.cos(pa)
    g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return (I / (2 * np.pi * sx * sy) * g).astype(np.float32)


def write_to_fits(radio_image: dict) -> Path:
    hdr = radio_image["wcs"].to_header()
    hdr["BUNIT"] = "Jy/pixel"
    hdu = fits.PrimaryHDU(data=radio_image["data"], header=hdr)
    path = Path(radio_image["path"])
    hdu.writeto(path, overwrite=True)
    return path


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", help="output path (output filename will be truth.fits).")
    common.add_argument("--osm", help= "path to osm skymodel input file.")
    common.add_argument("--npix", help= "pixel size per dimension (output image is square npix x npix).")
    common.add_argument("--dec0", default=-60.0, help="Phase center declinaison")
    common.add_argument("--ra0", default=-120.0, help="Phase center right ascenscion")
    common.add_argument("--fov-deg", type=float, default=1.0, help="angular field width [deg]")

    return common


def main(argv=None):

    args = build_parser().parse_args(argv)

    cell_arcsec =float(args.fov_deg) * 3600 /float(args.npix)

    out = Path(args.out,"truth.fits")
    osm_to_fits(args.osm, out, npix=int(args.npix), cell_arcsec=cell_arcsec, ra0=float(args.ra0), dec0=float(args.dec0) )


if __name__ == "__main__":
    sys.exit(main())

