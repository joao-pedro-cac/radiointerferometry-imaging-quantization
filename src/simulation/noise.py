"""
Heteroscedastic Gaussian noise injection into a CASA Measurement Set.

The Student-t noise model is a Gaussian scale mixture: each visibility i has
noise n_i ~ N(0, σ_i²). "Outliers" are visibilities where σ_i = scale * sigma0
instead of sigma0. This module injects that noise into the MS DATA column and
saves the ground truth σ_i array alongside the MS for EM weight validation.

Public API
----------
add_heteroscedastic_noise(ms_path, sigma0, epsilon, scale, seed)
    Reads DATA column, adds complex Gaussian noise with per-visibility variance,
    writes back. Saves obs_sigma.zarr in the same directory as the MS.
"""

from pathlib import Path

import dask
import dask.array as da
import numpy as np
import zarr
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_to_table


def add_heteroscedastic_noise(
    ms_path: str | Path,
    sigma0: float,
    epsilon: float,
    scale: float,
    seed: int,
) -> np.ndarray:
    """
    Add heteroscedastic complex Gaussian noise to the DATA column of an MS.

    For each (row, channel) visibility:
        σ_i = sigma0          with probability 1 - epsilon
        σ_i = scale * sigma0  with probability epsilon

    Noise added: n_i = N(0, σ_i²/2) + j·N(0, σ_i²/2), so E[|n_i|²] = σ_i².

    The WEIGHT column is set to 1/sigma0² uniformly — this is the imager's prior
    assumption (homoscedastic noise), which the EM algorithm then corrects.

    The ground truth σ_i array is saved as obs_sigma.zarr alongside the MS.

    Parameters
    ----------
    ms_path : path to the CASA MS directory
    sigma0 : baseline noise standard deviation in Jy
    epsilon : fraction of visibilities with inflated noise (outlier fraction)
    scale : σ_outlier = scale * sigma0
    seed : random seed for reproducibility

    Returns
    -------
    sigma : np.ndarray, shape (nrow_total, nchan)
        Ground truth per-visibility noise standard deviation.
    """
    ms_path = Path(ms_path)
    rng = np.random.default_rng(seed)

    datasets = list(xds_from_ms(str(ms_path), columns=["DATA", "WEIGHT"]))

    sigma_chunks = []
    new_datasets = []

    for ds in datasets:
        data = ds.DATA.data.compute()  # (nrow, nchan, ncorr)
        nrow, nchan, ncorr = data.shape

        # Per-(row, chan) noise standard deviation
        sigma = np.full((nrow, nchan), sigma0, dtype=np.float64)
        if epsilon > 0.0:
            outlier_mask = rng.random((nrow, nchan)) < epsilon
            sigma[outlier_mask] = scale * sigma0

        # Complex Gaussian noise: Re and Im each have std sigma/sqrt(2)
        std = sigma / np.sqrt(2.0)
        noise = (
            rng.standard_normal((nrow, nchan, ncorr))
            + 1j * rng.standard_normal((nrow, nchan, ncorr))
        ) * std[:, :, None]
        noise = noise.astype(data.dtype)

        new_data = data + noise

        # Uniform natural weight = 1/sigma0² (imager's starting assumption)
        new_weight = np.full((nrow, ncorr), 1.0 / sigma0**2, dtype=np.float64)

        new_ds = ds.assign({
            "DATA": (("row", "chan", "corr"), da.from_array(new_data, chunks=ds.DATA.data.chunks)),
            "WEIGHT": (("row", "corr"), da.from_array(new_weight, chunks=ds.WEIGHT.data.chunks)),
        })
        new_datasets.append(new_ds)
        sigma_chunks.append(sigma)

    writes = xds_to_table(new_datasets, str(ms_path), columns=["DATA", "WEIGHT"])
    dask.compute(writes)

    sigma_all = np.concatenate(sigma_chunks, axis=0)

    sigma_zarr_path = ms_path.parent / "obs_sigma.zarr"
    store = zarr.open_group(str(sigma_zarr_path), mode="w")
    chunks = (min(10_000, sigma_all.shape[0]), sigma_all.shape[1])
    if int(zarr.__version__.split(".")[0]) >= 3:
        store.create_array("sigma", shape=sigma_all.shape, dtype=sigma_all.dtype, chunks=chunks)
        store["sigma"][:] = sigma_all
    else:
        store.create_dataset("sigma", data=sigma_all, chunks=chunks)

    return sigma_all
