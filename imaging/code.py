from astropy.io import fits
import numpy as np
from image_data_analysis import *

# fits_data = fits.getdata("../../data/simulated/point_field_ctrl/truth_model.fits")
# fits_data = fits.getdata("../../data/simulated/point_field_ctrl/truth_model.fits")
# fits_data = np.squeeze(fits_data)

arr = np.random.randint(5, 35, (800, 500))
print(arr)
print(np.where(arr == 5, 1, 0))
print(np.count_nonzero(arr == 0))
create_histogram(arr, './teste.png')