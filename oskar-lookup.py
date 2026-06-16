import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ducc0.wgridder as wgrid
from time import time
from clean import hogbom_clean

# auxilary variables
BARSPACE = 90
VISIBILITY_MAXNUMBER = 4096
IMAGE_SIZE_X = 1 << 6
IMAGE_SIZE_Y = 1 << 6
LOOP_NUMBER = 1
SIMULATED_DATA_ABSOLUTEPATH = "../data/simulated/point_field_ctrl/obs_I.xds/ms0000_fid0000_spw0000_scan0000_band0000_time0000.zarr"


# data set simulated by OSKAR
oskar_simulated_dataset = xr.open_dataset(SIMULATED_DATA_ABSOLUTEPATH, engine="zarr")
print(oskar_simulated_dataset)
print()


# visibilities data subset
uvw    = oskar_simulated_dataset["UVW"]
freq   = oskar_simulated_dataset["FREQ"]
vis    = oskar_simulated_dataset["VIS"]
weight = oskar_simulated_dataset["WEIGHT"]

uvw_data = uvw.data
freq_data = freq.data
vis_data = np.squeeze(vis.data)
weight_data = np.squeeze(weight.data)


# image resolution parameters
npix_x, npix_y = IMAGE_SIZE_X, IMAGE_SIZE_Y    # image dimensions (in pixels)
pixel_size_x = np.radians(1.0) / IMAGE_SIZE_X  # 1 degree projected over N pixels (in projected radians)
pixel_size_y = np.radians(1.0) / IMAGE_SIZE_Y

time_start_dirty = time()

# dirty image computation
dirty_image = wgrid.vis2dirty(
    uvw=uvw_data,                       # uvw coordinates
    freq=freq_data,                     # channel frequencies
    vis=vis_data,                       # visibilities
    wgt=weight_data,                    # weight array
    npix_x=npix_x,                      # no. pixels in the x-axis
    npix_y=npix_y,                      # no. pixels in the y-axis
    pixsize_x=pixel_size_x,             # x-axis pixel size
    pixsize_y=pixel_size_y,             # y-axis pixel size
    epsilon=1e-5,                       # computation accuracy (>= 1e-5)
    do_wgridding=True,                  # perform the full algorithm
    nthreads=4,                         # no. threads used for computation
    double_precision_accumulation=True  # use double precision for computation
)

time_stop_dirty = time()

print(f"Image dirty shape: {dirty_image.shape}\n")
print(f"Computation time (dirty image): {time_stop_dirty - time_start_dirty} seconds")




time_start_psf = time()

# PSF computation
psf = wgrid.vis2dirty(
    uvw=uvw_data,                       # uvw coordinates
    freq=freq_data,                     # channel frequencies
    vis=np.ones_like(vis_data),         # visibilities
    wgt=weight_data,                    # weight array
    npix_x=npix_x,                      # no. pixels in the x-axis
    npix_y=npix_y,                      # no. pixels in the y-axis
    pixsize_x=pixel_size_x,             # x-axis pixel size
    pixsize_y=pixel_size_y,             # y-axis pixel size
    epsilon=1e-5,                       # computation accuracy (>= 1e-5)
    do_wgridding=True,                  # perform the full algorithm
    nthreads=4,                         # no. threads used for computation
    double_precision_accumulation=True  # use double precision for computation
)

time_stop_psf = time()

print(f"Computation time (PSF):         {time_stop_psf - time_start_psf} seconds\n")




time_start_clean = time()

# Run CLEAN on your existing arrays
model, residual, restored, components = hogbom_clean(
    dirty_image,  # from your vis2dirty call
    psf,          # from your PSF computation
    niter=10000,
    gain=0.1,
    threshold=np.std(dirty_image) * 0.005,  # 3 sigma threshold
    verbose=True
)

time_stop_clean = time()

print(f"\nComputation time (CLEAN):       {time_stop_clean - time_start_clean} seconds")












# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

im1 = axes[0, 0].imshow(dirty_image, cmap='inferno', origin='lower')
axes[0, 0].set_title('Dirty Image')

im2 = axes[0, 1].imshow(residual, cmap='inferno', origin='lower')
axes[0, 1].set_title(f'Residual')

im1 = axes[1, 0].imshow(model + 255, cmap='inferno', origin='lower')
axes[1, 0].set_title('Model')

im2 = axes[1, 1].imshow(restored, cmap='inferno', origin='lower')
axes[1, 1].set_title(f'Restored Image')

plt.tight_layout()
plt.show()



# # Compute PSF once (it's constant)
# psf = wgrid.vis2dirty(
#     uvw=uvw_data,
#     freq=freq_data,
#     vis=np.ones_like(vis_data),
#     wgt=weight_data,
#     npix_x=npix_x,
#     npix_y=npix_y,
#     pixsize_x=pixel_size_x,
#     pixsize_y=pixel_size_y,
#     epsilon=1e-5,
#     do_wgridding=True,
#     nthreads=4,
#     double_precision_accumulation=True
# )

# # Initial setup
# current_vis = vis_data.copy()
# current_dirty = dirty_image.copy()
# model_total = np.zeros_like(dirty_image)  # Accumulate model across iterations
# convergence_history = []  # Track improvement

# print(f"\nStarting iterative refinement with {LOOP_NUMBER} iterations...")
# print("=" * 60)

# for iteration in range(LOOP_NUMBER):
#     print(f"\n--- Iteration {iteration + 1}/{LOOP_NUMBER} ---")
    
#     # Step 1: CLEAN the current dirty image
#     model, residual, restored, components = hogbom_clean(
#         current_dirty,
#         psf,
#         niter=10000,
#         gain=0.1,
#         threshold=np.std(current_dirty) * 0.005,
#         verbose=True
#     )
    
#     # Step 2: Predict visibilities from the CLEAN model
#     predicted_vis = wgrid.dirty2vis(
#         uvw=uvw_data,
#         freq=freq_data,
#         dirty=model,  # Use the CLEAN model, not restored
#         wgt=np.ones_like(weight_data),
#         pixsize_x=pixel_size_x,
#         pixsize_y=pixel_size_y,
#         epsilon=1e-5,
#         do_wgridding=True,
#         vis=current_vis  # Initial guess
#     )
    
#     # Step 3: Calculate gain corrections (self-calibration)
#     # Compare observed vs predicted visibilities
#     # This is a simplified gain correction - in practice you'd solve per antenna
#     gain_amplitude = np.abs(current_vis) / (np.abs(predicted_vis) + 1e-10)
#     gain_phase = np.angle(current_vis) - np.angle(predicted_vis)
    
#     # Apply corrections (simplified - just use ratio)
#     corrected_vis = current_vis * (np.abs(current_vis) / (np.abs(predicted_vis) + 1e-10)) * \
#                     np.exp(1j * (np.angle(current_vis) - np.angle(predicted_vis)))
    
#     # Alternative: just use the predicted visibilities as new "observed"
#     # corrected_vis = predicted_vis  # This would be the simplest feedback
    
#     # Step 4: Generate new dirty image from corrected visibilities
#     current_dirty = wgrid.vis2dirty(
#         uvw=uvw_data,
#         freq=freq_data,
#         vis=corrected_vis,  # Use corrected visibilities
#         wgt=weight_data,
#         npix_x=npix_x,
#         npix_y=npix_y,
#         pixsize_x=pixel_size_x,
#         pixsize_y=pixel_size_y,
#         epsilon=1e-5,
#         do_wgridding=True,
#         nthreads=4,
#         double_precision_accumulation=True
#     )
    
#     # Step 5: Accumulate model
#     model_total += model
    
#     # Step 6: Check convergence
#     # Compute how much the dirty image changed
#     if iteration > 0:
#         change = np.std(current_dirty - previous_dirty) / np.std(current_dirty)
#         convergence_history.append(change)
#         print(f"Convergence metric (relative change): {change:.6f}")
        
#         # Early stopping if converged
#         if change < 1e-4:
#             print(f"Converged at iteration {iteration + 1}!")
#             break
    
#     # Store for next iteration
#     previous_dirty = current_dirty.copy()

# # Final restored image
# model_final, residual_final, restored_final, components_final = hogbom_clean(
#     current_dirty,
#     psf,
#     niter=10000,
#     gain=0.1,
#     threshold=np.std(current_dirty) * 0.005,
#     verbose=True
# )







# # Visualize results
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # Original dirty
# im1 = axes[0, 0].imshow(dirty_image, cmap='inferno', origin='lower')
# axes[0, 0].set_title('Original Dirty Image')
# plt.colorbar(im1, ax=axes[0, 0])

# # Final dirty after iterations
# im2 = axes[0, 1].imshow(current_dirty, cmap='inferno', origin='lower')
# axes[0, 1].set_title(f'Refined Dirty Image (iter {iteration+1})')
# plt.colorbar(im2, ax=axes[0, 1])

# # Final restored
# im3 = axes[0, 2].imshow(restored_final, cmap='inferno', origin='lower')
# axes[0, 2].set_title(f'Final Restored Image')
# plt.colorbar(im3, ax=axes[0, 2])

# # Difference (improvement)
# im4 = axes[1, 0].imshow(restored_final - dirty_image, cmap='RdBu', origin='lower')
# axes[1, 0].set_title('Improvement (Final - Original)')
# plt.colorbar(im4, ax=axes[1, 0])

# # Model accumulation
# im5 = axes[1, 1].imshow(model_total, cmap='inferno', origin='lower')
# axes[1, 1].set_title('Accumulated Model')
# plt.colorbar(im5, ax=axes[1, 1])

# # Convergence plot
# if convergence_history:
#     axes[1, 2].plot(convergence_history, 'b-o')
#     axes[1, 2].set_xlabel('Iteration')
#     axes[1, 2].set_ylabel('Relative Change')
#     axes[1, 2].set_title('Convergence')
#     axes[1, 2].grid(True)
# else:
#     # Show residual if no convergence history
#     im6 = axes[1, 2].imshow(residual_final, cmap='inferno', origin='lower')
#     axes[1, 2].set_title('Final Residual')
#     plt.colorbar(im6, ax=axes[1, 2])

# plt.tight_layout()
# plt.show()