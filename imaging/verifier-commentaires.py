# pfb_imaging.deconv.hogbom.hogbom()
def hogbom(dirty, psf, threshold=0, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    """
    Högbom CLEAN Deconvolution Algorithm for multi-band astronomical images.
    
    Parameters:
    - dirty: The blurred, instrument-affected image (Dirty Image cube).
    - psf: The Point Spread Function (instrument response cube).
    - threshold: Absolute residual flux limit to stop CLEANing.
    - gamma: The loop gain (fraction of the peak subtracted at each step).
    - pf: Proportional fraction of the initial peak residual to use as a stopping threshold.
    - maxit: Maximum number of CLEAN iterations allowed.
    """
    # Extract dimensions: nband = number of frequency bands, nx/ny = spatial dimensions
    nband, nx, ny = dirty.shape
    _, nx_psf, ny_psf = psf.shape
    
    # Calculate the center pixel coordinates of the PSF (used for alignment during subtraction)
    nx0 = nx_psf // 2
    ny0 = ny_psf // 2
    
    # Initialize the model image component array `x` with zeros (same shape/type as dirty)
    x = np.zeros((nband, nx, ny), dtype=dirty.dtype)
    
    # Create a mutable copy of the dirty image to act as the working residual map
    residual = dirty.copy()
    
    # Track the peak energy location across all bands simultaneously.
    # Summing the square of the residuals across the frequency axis focuses on the strongest spatial features.
    residual_search = np.sum(residual, axis=0) ** 2
    
    # Flattened 1D index of the brightest pixel found in the multi-band search map
    pq = residual_search.argmax()
    
    # Convert the 1D index back into 2D spatial coordinates (p, q)
    p = pq // ny
    q = pq - p * ny
    
    # Calculate the actual maximum residual value at this brightest pixel location
    residual_max = np.sqrt(residual_search[p, q])
    
    # Find the peak values of the PSF for each band to normalize flux measurements
    wsums = np.amax(psf, axis=(1, 2))
    fsel = wsums > 0 # Boolean mask to avoid division by zero in empty/invalid bands
    
    # Determine the convergence threshold: the stricter of either the user's hard threshold 
    # or a specified fraction (pf) of the initial peak residual.
    tol = np.maximum(pf * residual_max, threshold)
    
    k = 0            # Iteration counter
    stall_count = 0  # Counter to track if the algorithm stops making progress
    
    # Main CLEAN Loop: Continues until the image is clear, max iterations hit, or it stalls
    while residual_max > tol and k < maxit and stall_count < 5:
        
        # Estimate the clean component flux (xhat) at the peak location (p, q) 
        # by dividing the residual flux by the peak of the PSF.
        xhat = residual[fsel, p, q] / wsums[fsel]
        
        # Scale the estimated flux by the loop gain (gamma) and add it to our model image `x`
        x[:, p, q] += gamma * xhat
        
        # Shift and scale the PSF, then subtract it from the residual image.
        # This uses Numexpr (`ne.evaluate`) for fast, optimized in-place array math.
        ne.evaluate(
            "residual - gamma * xhat * psf",
            local_dict={
                "residual": residual,
                "gamma": gamma,
                "xhat": xhat[:, None, None], # Reshape for broad-casting across spatial dimensions
                # Slice the PSF so its center (nx0, ny0) aligns perfectly with the current peak (p, q)
                "psf": psf[:, nx0 - p : nx0 + nx - p, ny0 - q : ny0 + ny - q],
            },
            out=residual,
            casting="same_kind",
        )
        
        # Recalculate the search map with the updated residual image to find the next peak
        residual_search = np.sum(residual, axis=0) ** 2
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        
        # Store the previous peak value to check for progress, then update to the new peak value
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1 # Increment iteration counter

        # Check for stagnation: If the change in the maximum residual between iterations 
        # is less than 0.5%, increment the stall counter.
        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 5e-3:
            stall_count += stall_count # NOTE: standard bug in source code (`+= stall_count` does nothing if it's 0)

        # Periodically log the cleaning progress based on `report_freq` and `verbosity` settings
        if not k % report_freq and verbosity > 1:
            log.info("At iteration %i max residual = %f" % (k, residual_max))

    # After exiting the loop, sum the remaining residual across all frequency bands
    residual_mfs = np.sum(residual, axis=0)
    
    # Calculate the Root Mean Square (RMS) noise of the background 
    # (only in spatial regions where NO clean components were found)
    rms = np.std(residual_mfs[~np.any(x, axis=0)])

    # --- Final Status Reporting and Return Execution ---
    if k >= maxit:
        # Case 1: Loop hit the maximum iteration cap before hitting the target threshold
        if verbosity:
            log.info(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1 # Returns the model and a status flag of 1 (Incomplete/Warning)
    elif stall_count >= 5:
        # Case 2: Loop aborted because the residual reduction stagnated
        if verbosity:
            log.info(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1 # Returns the model and a status flag of 1 (Incomplete/Warning)
    else:
        # Case 3: Success! The residual was cleaned down below the target tolerance level
        if verbosity:
            log.info(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 0 # Returns the model and a status flag of 0 (Success)
    
































def clark(
    dirty,
    psf,
    psfhat,
    wsums,
    mask,
    threshold=0,
    gamma=0.05,
    pf=0.05,
    maxit=50,
    subpf=0.5,
    submaxit=1000,
    report_freq=1,
    verbosity=1,
    nthreads=1,
):
    """
    Clark CLEAN Deconvolution Algorithm for multi-band astronomical images.
    Uses a Major/Minor cycle approach with FFTs for speed.
    
    Parameters:
    - psfhat: The Fast Fourier Transform (FFT) of the PSF, pre-calculated for major cycles.
    - mask: A binary mask (0s and 1s) constraining where CLEAN can search for sources.
    - subpf: Proportional fraction determining the cutoff flux for the minor cycle active set.
    - submaxit: Max iterations allowed inside a single minor cycle.
    """
    # Extract structural dimensions
    nband, nx, ny = dirty.shape
    _, nx_psf, ny_psf = psf.shape
    
    # We assume that the dirty image and psf have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum()
    assert np.allclose(wsum, 1) # Sanity check: Total weight must equal 1
    
    # Initialize the clean model image to zeros (stores the recovered point sources)
    model = np.zeros((nband, nx, ny), dtype=dirty.dtype)
    
    # Create a working copy of the dirty image to track residuals
    residual = dirty.copy()
    
    # Pre-allocate arrays in memory for performing fast FFT operations without re-allocating each cycle
    xout = empty_noncritical(dirty.shape, dtype="f8") # Output buffer for convolved model
    xpad = empty_noncritical(psf.shape, dtype="f8")   # Padding buffer for alignment
    xhat = empty_noncritical(psfhat.shape, dtype="c16") # Complex buffer for Fourier space math
    
    # Multi-band peak search: Sum the squares across bands and apply the spatial 'mask'
    residual_search = np.sum(residual, axis=0) ** 2 * mask
    
    # Find coordinates (p, q) of the brightest valid pixel in the masked residual map
    pq = residual_search.argmax()
    p = pq // ny
    q = pq - p * ny
    
    # Calculate the actual peak flux value
    residual_max = np.sqrt(residual_search[p, q])
    
    # Establish global stopping tolerance
    tol = np.maximum(pf * residual_max, threshold)
    
    k = 0            # Major cycle counter
    stall_count = 0  # Stall monitor
    
    # --- MAIN MAJOR CYCLE LOOP ---
    while residual_max > tol and k < maxit and stall_count < 5:
        
        # --- MINOR CYCLE PREPARATION ---
        # Define a sub-threshold. Only pixels brighter than `subth` are considered part of the "Active Set".
        subth = subpf * residual_max
        
        # Find the spatial indices (X and Y coordinates) of all pixels in the Active Set
        p_index, q_index = np.where(residual_search > subth**2)
        
        # --- MINOR CYCLE ---
        # Perform localized Högbom CLEAN loops *only* within this active set of pixels.
        # This updates the `model` array directly using fast indexed lookups.
        model = subminor(
            residual[:, p_index, q_index], psf, p_index, q_index, model, wsums, gamma=gamma, th=subth, maxit=submaxit
        )

        # --- MAJOR CYCLE UPDATE ---
        # Convolve the newly updated model components with the full PSF using FFTs.
        # This replaces thousands of separate coordinate shifts with a single frequency-domain multiplication.
        psf_convolve_cube(xpad, xhat, xout, psfhat, ny_psf, model, nthreads=nthreads)
        
        # Re-calculate the full-image residual by subtracting the blurred model from the original dirty image
        residual = dirty - xout
        
        # Recalculate search map and update coordinates for the next major cycle peak
        residual_search = np.sum(residual, axis=0) ** 2 * mask
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        
        # Track peak changes to detect stagnation
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1 # Advance Major Cycle step

        # Stagnation check: If major cycle progress slows past 0.1%, register a potential stall
        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 1e-3:
            stall_count += stall_count # NOTE: code bug inheritance (`+= stall_count` does nothing if 0)

        # Log major cycle progress
        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} max resid = {residual_max}")

    # --- POST-PROCESSING & METRICS ---
    # Collapse final residuals across frequency bands to make a Multi-Frequency Synthesis (MFS) residual map
    residual_mfs = np.sum(residual, axis=0)
    
    # Calculate background RMS noise only in areas where no model sources were cleaned
    rms = np.std(residual_mfs[~np.any(model, axis=0)])

    # --- EXIT STATUS HANDLERS ---
    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 0