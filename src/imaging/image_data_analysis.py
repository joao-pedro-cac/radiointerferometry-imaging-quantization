import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.convolution import convolve_fft
from astropy.io import fits

# create a bitlength frequency histogram
def create_bitlength_histogram(array, filepath, histogram_title):
    for i in range(len(array)):
        array[i] = int.bit_length(array[i])

    uniques = np.unique(array)

    max_unique = np.max(uniques)

    horizontal_axe = np.arange(0, max_unique + 1)
    vertical_axe = [int(np.count_nonzero(array == i)) for i in horizontal_axe]

    plt.figure()
    plt.bar(horizontal_axe, vertical_axe)

    plt.title(histogram_title)
    plt.xlabel("Number of encoded bits")
    plt.ylabel("Counting frequency")
    
    plt.savefig(filepath, format="png")
    plt.close()

def save_restored_fits(clean_model, psf, out_path, template_header_path=None,
                       residual=None, orient=True):
    """Restore = clean_model (*) fitted clean beam [+ residual], write FITS (Jy/beam).

    clean_model : de-scaled model, SKY units, SKY orientation (pre transpose/flip).
    psf         : PSF in the same (un-transposed) orientation. Amplitude scale
                  irrelevant — the Gaussian FWHM fit is amplitude-invariant, so a
                  quantized/undescaled PSF is fine here.
    residual    : optional, MUST already be in clean_model's sky units. Do NOT pass
                  the raw residual_dirty_image (it's still on scalefactor_dirty).
    orient      : apply transpose+flip so the frame matches clean_model.fits.
    """
    bmaj_px, bmin_px, pa_rad = synth_beam_from_psf(psf)
    kern = make_beam_kernel(bmaj_px, bmin_px, pa_rad)          # peak-normalized -> Jy/beam
    restored = restore(np.asarray(clean_model, np.float64), kern)
    if residual is not None:
        restored = restored + np.asarray(residual, np.float64)

    header, cdelt = None, None
    if template_header_path is not None:
        try:
            with fits.open(template_header_path) as hdul:
                header = hdul[0].header.copy()
            cdelt = abs(float(header.get("CDELT2", 0.0))) or None
        except FileNotFoundError:
            header = None

    if orient:                                                # match clean_model.fits frame
        restored = np.transpose(restored)[::-1, :]

    hdu = fits.PrimaryHDU(data=restored.astype(np.float32), header=header)
    hdu.header["BUNIT"] = "Jy/beam"
    if cdelt:                                                 # pixel FWHM -> deg for beam keywords
        hdu.header["BMAJ"] = bmaj_px * cdelt
        hdu.header["BMIN"] = bmin_px * cdelt
        hdu.header["BPA"]  = np.degrees(pa_rad)               # NB: sky-frame PA, see caveat
        hdu.header["BLANK"] = " restored-image beam (fitted from PSF)"
    hdu.writeto(out_path, overwrite=True)
    return bmaj_px, bmin_px, pa_rad

# ---------------------------------------------------------------------------
# clean-beam preprocessing: fit the PSF main lobe, restore to common resolution
# ---------------------------------------------------------------------------
def synth_beam_from_psf(psf, fit_halfwidth_px=32):
    """Fit a 2-D Gaussian to the PSF main lobe -> (bmaj_px, bmin_px, pa_rad) FWHMs.
    This IS the clean/restoring beam; fit on a cutout so sidelobes don't drag it."""
    yx = np.unravel_index(np.argmax(psf), psf.shape)
    h = fit_halfwidth_px
    sl = (slice(max(yx[0]-h, 0), yx[0]+h+1), slice(max(yx[1]-h, 0), yx[1]+h+1))
    cut = np.asarray(psf[sl], dtype=np.float64)
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    p0 = models.Gaussian2D(cut.max(), cut.shape[1]//2, cut.shape[0]//2, 2.0, 2.0)
    g = fitting.LevMarLSQFitter()(p0, xx, yy, cut)
    f = 2.0 * np.sqrt(2.0 * np.log(2.0))                      # sigma -> FWHM
    sx, sy = abs(g.x_stddev.value), abs(g.y_stddev.value)
    return f*max(sx, sy), f*min(sx, sy), float(g.theta.value)

def make_beam_kernel(bmaj_px, bmin_px, pa_rad, size=None, unit_sum=False):
    s = size or int(2*np.ceil(2*bmaj_px) + 1)
    yy, xx = np.mgrid[:s, :s] - s//2
    sx, sy = bmaj_px/2.3548, bmin_px/2.3548
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    xr, yr = xx*ct + yy*st, -xx*st + yy*ct
    k = np.exp(-0.5*((xr/sx)**2 + (yr/sy)**2))
    return k/k.sum() if unit_sum else k   

def restore(image, kernel):
    return convolve_fft(np.asarray(image, np.float64), kernel,
                        normalize_kernel=False, allow_huge=True)

def support_mask(true_image, tol=1e-4):
    t = np.asarray(true_image, np.float64)
    return t > tol*t.max()

def faint_mask(true_image, faint_frac=1e-2, tol=1e-6):
    t = np.asarray(true_image, np.float64)
    return (t > tol*t.max()) & (t < faint_frac*t.max())


def compute_rms(image):                                       # image RMS (quadratic mean)
    x = np.asarray(image, np.float64)
    return np.sqrt(np.mean(x*x))

def compute_achieved_dr(residual_image, peak, eps=1e-12):
    """Achieved DR = peak / off-source residual RMS. MAD-based RMS so residual
    source flux doesn't inflate the floor. Linear ratio (not dB)."""
    r = np.asarray(residual_image, np.float64).ravel()
    sigma = 1.4826*np.median(np.abs(r - np.median(r)))
    return peak/(sigma + eps)

def compute_snr(restored, true_restored, eps=1e-12):
    """SNR vs truth (dB). Truth power in numerator; peak-dominated (MSE)."""
    a = np.asarray(restored, np.float64); b = np.asarray(true_restored, np.float64)
    mse = np.mean((a - b)**2)
    return 10*np.log10(np.mean(b*b)/(mse + eps))

def compute_psnr(restored, true_restored, eps=1e-12):
    """PSNR vs truth (dB). Peak² over MSE; peak-dominated."""
    a = np.asarray(restored, np.float64); b = np.asarray(true_restored, np.float64)
    mse = np.mean((a - b)**2)
    return 10*np.log10(np.max(b)**2/(mse + eps))

def compute_cross_correlation(restored, true_restored):
    """Zero-lag Pearson correlation in [-1,1]. Scale+offset invariant ->
    blind to flux-scale error and uniform faint loss (pair with flux_ratio)."""
    x = np.asarray(restored, np.float64).ravel()
    y = np.asarray(true_restored, np.float64).ravel()
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt(np.dot(x, x)*np.dot(y, y))
    return float("nan") if denom == 0.0 else np.dot(x, y)/denom

def compute_flux_ratio(model_raw, true_raw, eps=1e-12):
    """Grid-agnostic flux conservation. Peak-normalized totals so different
    pixel counts (PFB model grid vs OSKAR truth grid) are comparable."""
    a = np.asarray(model_raw, np.float64)
    b = np.asarray(true_raw, np.float64)
    return float((a.sum()/(a.max()+eps)) / (b.sum()/(b.max()+eps) + eps))


# compute the SSIM of an image with respect to another image
def compute_ssim(clean_image, original_image):
    # averages
    clean_image_avg = np.average(clean_image)
    original_image_avg = np.average(original_image)

    # variances
    clean_image_variance = np.var(clean_image)
    original_image_variance = np.var(original_image)

    # standard deviations
    clean_image_std = np.sqrt(clean_image_variance)
    original_image_std = np.sqrt(original_image_variance)

    # covariance between both images
    covariance = np.sum((clean_image - clean_image_avg) * (original_image - original_image_avg)) / clean_image.size


    # computation auxiliary variables
    image_numbits = original_image.dtype.alignment * 8
    L = 2 ** image_numbits - 1
    k1 = 0.01
    k2 = 0.03

    # stabilization variables
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2

    # computation components
    l = (2 * clean_image_avg * original_image_avg + c1) / (clean_image_avg ** 2 + original_image_avg ** 2 + c1)
    c = (2 * clean_image_std * original_image_std + c2) / (clean_image_variance + original_image_variance + c2)
    s = (covariance + c3) / (clean_image_std * original_image_std + c3)

    return l * c * s
