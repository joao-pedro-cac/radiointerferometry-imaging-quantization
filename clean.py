import numpy as np
from scipy.ndimage import gaussian_filter

def hogbom_clean(dirty, psf, niter=999, gain=0.1, threshold=None, verbose=True):
    """
    Hogbom CLEAN algorithm - works directly with your arrays
    
    Parameters:
    -----------
    dirty : 2D array - Dirty image
    psf : 2D array - Point Spread Function (dirty beam)
    niter : int - Maximum number of iterations
    gain : float - Loop gain (0.05-0.2 typical)
    threshold : float - Stop when peak < threshold
    verbose : bool - Print progress
    
    Returns:
    --------
    model : 2D array - CLEAN component model
    residual : 2D array - Residual image after deconvolution
    restored : 2D array - Final restored image
    """
    psf_norm = psf / np.max(psf)  # normalize PSF to peak = 1
    
    # find PSF peak coordinates
    psf_peak = np.unravel_index(np.argmax(psf_norm), psf_norm.shape)
    
    # initialize
    residual = dirty.copy()
    model = np.zeros_like(dirty)
    
    # set threshold if not provided (default: 1% of max dirty)
    if threshold is None:
        threshold = 0.01 * np.max(np.abs(dirty))
    
    components = []
    rms_init = np.std(dirty)
    
    if verbose:
        print(f"Starting CLEAN: {niter} max iterations, gain={gain}, threshold={threshold:.6f}")
        print(f"Initial RMS: {rms_init:.6f}")
    
    for i in range(niter):
        # find peak in residual
        peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak_flux = residual[peak_idx]
        
        # check stopping condition
        if np.abs(peak_flux) < threshold:
            if verbose:
                print(f"Stopped at iteration {i} \
                      |peak| = {np.abs(peak_flux):.6f} < {threshold}")
            break
        
        # add component to model
        component_flux = peak_flux * gain
        model[peak_idx] += component_flux
        
        # subtract shifted PSF from residual
        y_shift = peak_idx[0] - psf_peak[0]
        x_shift = peak_idx[1] - psf_peak[1]
        shifted_psf = np.roll(np.roll(psf_norm, y_shift, axis=0), x_shift, axis=1)
        residual -= component_flux * shifted_psf
        
        components.append((peak_idx, component_flux))
        
        if verbose and (i + 1) % 100 == 0:
            current_rms = np.std(residual)
            print(f"  Iteration {i+1}: peak={peak_flux:.6f}, RMS={current_rms:.6f}, components={len(components)}")
    
    if verbose:
        print(f"CLEAN finished: {len(components)} components extracted")
        print(f"Final RMS: {np.std(residual):.6f}")
    
    # restoration: convolve model with Gaussian clean beam + residual
    
    # estimate clean beam FWHM from PSF main lobe
    y_peak, x_peak = psf_peak
    y_profile = psf_norm[:, x_peak]
    x_profile = psf_norm[y_peak, :]
    
    # find FWHM (half maximum)
    half_max = 0.5
    y_above = np.where(y_profile >= half_max)[0]
    x_above = np.where(x_profile >= half_max)[0]
    
    if len(y_above) > 1 and len(x_above) > 1:
        fwhm_y = (y_above[-1] - y_above[0])
        fwhm_x = (x_above[-1] - x_above[0])
        sigma = max(fwhm_y, fwhm_x) / (2 * np.sqrt(2 * np.log(2)))
    else:
        sigma = 2.0  # Default guess
    
    if verbose:
        print(f"Clean beam sigma = {sigma:.2f} pixels")
    
    # apply Gaussian convolution to model
    restored = gaussian_filter(model, sigma=sigma) + residual
    
    return model, residual, restored, components