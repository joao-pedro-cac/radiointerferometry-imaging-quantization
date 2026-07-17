import numpy as np
import matplotlib.pyplot as plt

# create a bitlength frequency histogram
def create_bitlength_histogram(array, filepath):
    uniques = np.unique(array)

    min_unique = np.min(uniques)
    max_unique = np.max(uniques)

    horizontal_axe = np.arange(min(min_unique, 0), max_unique + 1)
    vertical_axe = [int(np.count_nonzero(array == i)) for i in horizontal_axe]

    plt.figure()
    plt.bar(horizontal_axe, vertical_axe)

    plt.title(f"Bit encoding frequency")
    plt.xlabel("Number of encoded bits")
    plt.ylabel("Counting frequency")
    
    plt.savefig(filepath)
    plt.close()


# compute the root-mean-square value of an image
def compute_rms(image):
    power = image ** 2
    average_power = np.sum(power) / np.size(power)
    rms = np.sqrt(average_power)

    return rms


# compute the dynamic range (in decibels) of an image
def compute_dr(image, eps=1e-12):
    image = np.abs(image).flatten()

    try:
        min_value = np.min(image[image > 0])
        max_value = np.max(image)

        return 20 * np.log10(max_value / (min_value + eps))                 # unit conversion to decibels
    except:
        return -1                                                           # error code


# compute the SNR of an image with respect to another image
def compute_snr(clean_image, original_image, eps=1e-12):
    noise_power = np.sum((clean_image - original_image) ** 2)
    mse = noise_power / clean_image.size

    clean_image_power = np.sum(clean_image ** 2)

    return 10 * np.log10(clean_image_power / (mse + eps))                   # unit conversion to decibels


# compute the PSNR of an image with respect to another image
def compute_psnr(clean_image, original_image, eps=1e-12):
    noise_power = np.sum((clean_image - original_image) ** 2)
    mse = noise_power / clean_image.size

    return 10 * np.log10((np.max(clean_image) ** 2) / (mse + eps))                 # unit conversion to decibels


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
