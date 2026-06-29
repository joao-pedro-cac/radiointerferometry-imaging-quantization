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
    restored_noresidual = gaussian_filter(model, sigma=sigma)
    restored = restored_noresidual + residual
    
    return model, residual, restored, restored_noresidual, components






































import numexpr as ne
import numpy as np


def hogbom_2d(dirty, psf, threshold=0, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    """
    Högbom CLEAN Deconvolution Algorithm for 2D astronomical images.
    
    Parameters:
    - dirty: The blurred, instrument-affected 2D image.
    - psf: The Point Spread Function (2D instrument response).
    - threshold: Absolute residual flux limit to stop CLEANing.
    - gamma: The loop gain (fraction of the peak subtracted at each step).
    - pf: Proportional fraction of the initial peak residual to use as a stopping threshold.
    - maxit: Maximum number of CLEAN iterations allowed.
    """
    # Extract 2D spatial dimensions
    nx, ny = dirty.shape
    nx_psf, ny_psf = psf.shape
    
    # Calculate the center pixel coordinates of the PSF
    nx0 = nx_psf // 2
    ny0 = ny_psf // 2
    
    # Initialize the model image `x` with zeros
    x = np.zeros((nx, ny), dtype=dirty.dtype)
    
    # Create a mutable copy of the dirty image to act as the working residual map
    residual = dirty.copy()
    
    # Square the residuals to find the peak energy location (handles both pos/neg peaks)
    residual_search = residual ** 2
    
    # Flattened 1D index of the brightest pixel
    pq = residual_search.argmax()
    
    # Convert the 1D index back into 2D spatial coordinates (p, q)
    p = pq // ny
    q = pq - p * ny
    
    # Calculate the actual maximum residual value at this brightest pixel location
    residual_max = np.sqrt(residual_search[p, q])
    
    # Find the peak value of the PSF to normalize flux measurements
    wsums = np.amax(psf)
    if wsums <= 0:
        raise ValueError("PSF peak must be greater than zero.")
    
    # Determine the convergence threshold
    tol = np.maximum(pf * residual_max, threshold)
    
    k = 0            # Iteration counter
    stall_count = 0  # Counter to track if the algorithm stops making progress
    
    # Main CLEAN Loop
    while residual_max > tol and k < maxit and stall_count < 5:
        
        # Estimate the clean component flux (xhat) at the peak location (p, q)
        xhat = residual[p, q] / wsums
        
        # Scale the estimated flux by the loop gain (gamma) and add it to our model image `x`
        x[p, q] += gamma * xhat
        
        # Shift and scale the 2D PSF, then subtract it from the residual image
        # Calculate the shift needed to align the PSF peak (nx0, ny0) to the residual peak (p, q)
        y_shift = p - nx0
        x_shift = q - ny0
        shifted_psf = np.roll(np.roll(psf, y_shift, axis=0), x_shift, axis=1)

        ne.evaluate(
            "residual - gamma * xhat * psf",
            local_dict={
                "residual": residual,
                "gamma": gamma,
                "xhat": xhat,  
                "psf": shifted_psf, # Use the correctly rolled/shifted PSF
            },
            out=residual,
            casting="same_kind",
        )
        
        # Recalculate the search map with the updated residual image to find the next peak
        residual_search = residual ** 2
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        
        # Store the previous peak value to check for progress, then update to the new peak value
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1 

        # Check for stagnation: If the change in the maximum residual is less than 0.5%
        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 5e-3:
            stall_count += 1  # Fixed: Changed from `+= stall_count` to properly increment
        else:
            stall_count = 0   # Reset stall counter if progress is being made

        # Periodically log progress
        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} max residual = {residual_max}")

    # Calculate Root Mean Square (RMS) noise where NO clean components were found
    rms = np.std(residual[x == 0])

    # --- Final Status Reporting and Return Execution ---
    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 0




























def hogbom(dirty_image=None, psf=None, niter_max=1e3, gain=0.1, threshold=0.1, verbose=True):
    """
    Hogbom CLEAN Deconvolution Algorithm
    ------------------------------------

    Deconvolves a `dirty_image` by using the ``Hogbom`` variant of the CLEAN algorithm.

    Parameters
    ----------
    dirty_image : array_like
                  Input array.
    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0, 1, ..., N-1] where N is the number of axes of `a`. Negative
        indices can also be used to specify axes. The i-th axis of the returned
        array will correspond to the axis numbered ``axes[i]`` of the input.
        If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses
        the order of the axes.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    See Also
    --------
    ndarray.transpose : Equivalent method.
    moveaxis : Move axes of an array to new positions.
    argsort : Return the indices that would sort an array.

    Notes
    -----
    Use ``transpose(a, argsort(axes))`` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> np.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> np.transpose(a)
    array([1, 2, 3, 4])

    >>> a = np.ones((1, 2, 3))
    >>> np.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)

    >>> a = np.ones((2, 3, 4, 5))
    >>> np.transpose(a).shape
    (5, 4, 3, 2)

    >>> a = np.arange(3*4*5).reshape((3, 4, 5))
    >>> np.transpose(a, (-1, 0, -2)).shape
    (5, 3, 4)

    """
    # 1. create a residual image (copy of dirty image)
    residual_image = dirty_image.copy()

    # 2. locate max brightness in residual image
    val_max = np.max(residual_image)
    l_max, m_max = np.where(residual_image == val_max)

    # type casting
    l_max = int(l_max[0])
    m_max = int(m_max[0])

    if verbose:
        print("Max point computed")
        print(f"Coordinates: ({l_max}, {m_max})")
        print(f"Value:       {val_max}")

    # 3. compute the PSF centered at (Lmax, Mmax)


    # 4. subtract the residual image by a fraction of PSFmax at (Lmax, Mmax)
    # 5. insert a scaled Dirac at (Lmax, Mmax) in the model image
    # 6. go back to step 2 until max(residual image) < threshold
    # 7. convolve the Diracs in the model image with a clean-beam response (a Gaussian)
    # 8. final image = model image + residual image