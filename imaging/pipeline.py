"""
ImagePipeline
===============

A class that implements an interferometry-based 2D image processing pipeline.

Attributes
----------

- `clean_algorithm` Current CLEAN algorithm variant adopted by the pipeline (default: Högbom)
- `gridding_epsilon` Epsilon parameter used in the gridding stage: It must be at least 1E-5 (default: 1E-5)
- `clean_gamma` Gamma parameter of the CLEAN algorithm (default: 0.001)
- `clean_pf` Peak fraction parameter of the CLEAN algorithm (default: 0.005)
- `clean_maxiter` Maximum number of iterations allowed for the CLEAN algorithm in the deconvolution stage (default: 1E3)
- `true_image` File path of the FITS file containing the true (or simulated) sky model

Methods
-------

- `set_true_model` Set the file path of the of the FITS file containing the true (or simulated) sky model. It must be used to assign a true sky model to the pipeline object before any other function is implemented
- `set_clean_algorithm` Set the CLEAN algorithm variant to be used
- `compute_dirty_image` Compute the dirty image from a set of visibilities
- `compute_psf` Compute the point spread function (PSF) from a set of visibilities
- `compute_clean_image` Compute the clean model from a (dirty image, PSF) pair and the computation status (0 for a complete deconvolution, 1 for error or maximum number of iterations reached)
- `compute_visibilities` Compute a set of visibilities from an image
- `calculate_image_metrics` Compute the RMS, DR, SNR, PSNR and SSIM of an image based on the true sky model

"""

import numpy as np
import ducc0.wgridder
import pfb_imaging.deconv.clark as pfb_clark
import pfb_imaging.deconv.hogbom as pfb_hogbom
import imaging.image_data_analysis as image_data_analysis
from astropy.io import fits
from misc.fileread import *


class ImagePipeline():
    def __init__(self,
                 clean_algorithm="hogbom",
                 gridding_epsilon=1e-5,
                 clean_gamma=0.001,
                 clean_pf=0.005,
                 clean_maxiter=1e3):
        assert gridding_epsilon >= 1e-5         # minimum accuracy
        assert clean_algorithm in ["hogbom", "clark"]

        self.clean_algorithm = clean_algorithm
        self.gridding_epsilon = gridding_epsilon
        self.clean_gamma = clean_gamma
        self.clean_pf = clean_pf
        self.clean_maxiter = clean_maxiter


    # image resolution parameters
    def set_true_model(self, true_model_path):
        self.true_image = np.squeeze(fits.getdata(true_model_path))    # true (original) sky image
        self.__npix_x, self.__npix_y, self.__pixel_size_x, self.__pixel_size_y = extract_pixel_info(true_model_path)
        self.__pixel_size_x = np.radians(self.__pixel_size_x)            # degrees to radians conversion
        self.__pixel_size_y = np.radians(self.__pixel_size_y)

    
    # CLEAN algorithm setup
    def set_clean_algorithm(self, clean_algorithm):
        assert clean_algorithm in ["hogbom", "clark"]
        self.clean_algorithm = clean_algorithm


    # dirty image computation
    def compute_dirty_image(self, uvw, freq, vis, wgt, verbosity=False):
        uvw = uvw.astype("float64")             # it must be float64
        freq = freq.astype("float64")           # it must be float64

        dirty_image = ducc0.wgridder.vis2dirty(
            uvw=uvw,                            # uvw coordinates
            freq=freq,                          # channel frequencies
            vis=vis,                            # visibilities
            wgt=wgt,                            # weight array_dirty_image
            npix_x=self.__npix_x,               # no. pixels in the x-axis
            npix_y=self.__npix_y,               # no. pixels in the y-axis
            pixsize_x=self.__pixel_size_x,      # x-axis pixel size
            pixsize_y=self.__pixel_size_y,      # y-axis pixel size
            epsilon=self.gridding_epsilon,      # computation accuracy
            do_wgridding=True,                  # perform the full algorithm
            nthreads=8,                         # no. threads used for computation
            double_precision_accumulation=True, # use double precision for computation
            verbosity=verbosity
        )

        return dirty_image


    # PSF computation
    def compute_psf(self, uvw, freq, vis, wgt, verbosity=False):
        uvw = uvw.astype("float64")             # it must be float64
        freq = freq.astype("float64")           # it must be float64

        psf = ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=np.ones_like(vis),
            wgt=wgt,
            npix_x=2 * self.__npix_x,           # 2x grid size (to handle edge padding in CLEAN)
            npix_y=2 * self.__npix_y,           # 2x grid size (to handle edge padding in CLEAN)
            pixsize_x=self.__pixel_size_x,             
            pixsize_y=self.__pixel_size_y,
            epsilon=self.gridding_epsilon,
            do_wgridding=True,
            nthreads=8,
            double_precision_accumulation=True,
            verbosity=verbosity
        )

        return psf


    # CLEAN computation
    def compute_clean_image(self, dirty_image, psf, verbosity=True):
        if psf.dtype != dirty_image.dtype:
            psf = psf.astype(dirty_image.dtype)

        if self.clean_algorithm == "hogbom":          # Hogbom CLEAN
            model_cube, status = pfb_hogbom.hogbom(
                dirty_image[np.newaxis, :, :],        # new axis to match three dimensions
                psf[np.newaxis, :, :],                # new axis to match three dimensions
                threshold=0,                          # force it to rely on the loop fractional threshold (pf)
                gamma=self.clean_gamma,               # loop gain
                pf=self.clean_pf,                     # stop when max residual drops to (pf * peak_dirty_value)
                maxit=self.clean_maxiter,
                verbosity=verbosity
            )
        else:                                         # Clark CLEAN
            # normalize the dirty image and the PSF to Jy/beam
            psf_peak = np.max(psf)
            dirty_image /= psf_peak
            psf /= psf_peak

            # prepare the 3D representations for dirty image and PSF
            dirty_3d = dirty_image[np.newaxis, :, :]
            psf_3d = psf[np.newaxis, :, :]

            # generate the mandatory parameters required by Clark
            wsums = np.array([1.0], dtype=dirty_image.dtype)
            mask = np.ones(dirty_image.shape, dtype=dirty_image.dtype)

            # Compute the Fourier transform of the PSF (psfhat)
            # Note: In standard pipelines, the PSF must be shifted to the origin before the FFT
            # to prevent phase ramps during convolution.
            psf_3d_shifted = np.fft.ifftshift(psf_3d, axes=(1, 2))
            psfhat = np.fft.rfft2(psf_3d_shifted, axes=(1, 2))

            model_cube, status = pfb_clark.clark(
                dirty=dirty_3d, 
                psf=psf_3d, 
                psfhat=psfhat,
                wsums=wsums,
                mask=mask,
                threshold=0,                          # Force reliance on fractional threshold (pf)
                gamma=self.clean_gamma,               # Default loop gain for Clark (can be adjusted)
                pf=self.clean_pf,                     # Stop when max residual drops to 0.1% of peak
                maxit=self.clean_maxiter,             # Major cycles max limit
                subpf=0.5,                            # Subminor loop threshold fraction
                submaxit=1000,                        # Max iterations per subminor loop
                nthreads=8,                           # Match your wgridder thread count
                verbosity=verbosity
            )

        model = np.squeeze(model_cube)                # extract 2D model

        return model, status


    # visibilities reconstruction
    def compute_visibilities(self, dirty_image, uvw, freq, wgt, verbosity=False):
        uvw = uvw.astype("float64")             # it must be float64
        freq = freq.astype("float64")           # it must be float64

        vis = ducc0.wgridder.dirty2vis(
            uvw=uvw,                            # uvw coordinates
            freq=freq,                          # channel frequencies
            dirty=dirty_image,                  # dirty image
            wgt=wgt,                            # weight array
            pixsize_x=self.__pixel_size_x,      # x-axis pixel size
            pixsize_y=self.__pixel_size_y,      # y-axis pixel size
            center_x=0,                         # x-coordinate of the center relative to the phase center
            center_y=0,                         # y-coordinate of the center relative to the phase center
            epsilon=self.gridding_epsilon,      # computation accuracy (>= 1e-5)
            do_wgridding=True,                  # perform the full algorithm
            nthreads=8,                         # no. threads used for computation    
            verbosity=verbosity
        )

        return vis
    

    # image metrics calculation
    def calculate_image_metrics(self, clean_image):
        image_metrics = {
            "rms"           : float(image_data_analysis.compute_rms(clean_image)),
            "dynamic_range" : float(image_data_analysis.compute_dr(clean_image)),
            "snr"           : float(image_data_analysis.compute_snr(clean_image, self.true_image)),
            "psnr"          : float(image_data_analysis.compute_psnr(clean_image, self.true_image)),
            "ssim"          : float(image_data_analysis.compute_ssim(clean_image, self.true_image))
        }

        return image_metrics