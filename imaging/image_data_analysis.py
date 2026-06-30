import numpy as np
import matplotlib.pyplot as plt

def create_histogram(array, filepath, xlog=False, ylog=False):
    array = array.flatten()

    uniques, count = np.unique(array, return_counts=True)

    plt.figure()

    if xlog:
        uniques += np.abs(np.min(uniques)) + 1
        plt.xscale('log')
    if ylog:
        count += np.abs(np.min(count)) + 1
        plt.yscale('log')

    plt.bar(uniques, count)

    plt.title(f"Data Frequency Histogram")
    plt.xlabel("Value" + (" (log)" if xlog else ""))
    plt.ylabel("Counting frequency" + (" (log)" if ylog else ""))
    
    plt.savefig(filepath)
    plt.close()

# def compute_psnr()