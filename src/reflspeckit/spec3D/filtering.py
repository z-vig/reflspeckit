# Dependencies
import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d

# Relative Imports
from .polyfit import polyfit_cube


def box_filter_cube(
    raw_cube: np.ndarray, window_size: int = 5
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Smoothes a spectrum using a moving average.

    Parameters
    ----------
    original_spectrum: np.ndarray
        Non-smooth spectral cube.
    window_size: optional, int
        Window size to use for the moving average. Default is 5.
    edge_handling: optional, str
        How to handle the edge cases of spectrum after convolution. Default is
        `"extrapolate"`
    remove_outliers: optional, bool
        If true, outliers are removed from the spectrum. A threshold value
        can be passed. Default is False.
    outlier_threshold: optional, float
        Sigma threshold to be used in outlier removal. See
        `outlier_removal.py`. Default is 2.
    """
    cube: npt.NDArray = raw_cube

    window = np.ones(window_size)
    nbands = cube.shape[2]

    edge_length = int(np.maximum(round(cube.shape[2] * 0.1, 0), 2))

    left_idx: npt.NDArray[np.int32] = np.arange(
        0, 1 + edge_length, dtype=np.int32
    )
    right_idx: npt.NDArray[np.int32] = np.arange(
        nbands - edge_length, nbands, dtype=np.int32
    )

    left_fit = polyfit_cube(cube[:, :, left_idx], left_idx, 1)
    left_ext = np.arange(-window_size, 0)
    left_arm = np.einsum(
        "ij,...j->...i", np.vander(left_ext, 2), left_fit.beta
    )

    right_fit = polyfit_cube(cube[:, :, right_idx], right_idx, 1)
    right_ext = np.arange(nbands, nbands + window_size)
    right_arm = np.einsum(
        "ij,...j->...i", np.vander(right_ext, 2), right_fit.beta
    )

    smoothed_spectrum = np.empty(
        (*cube.shape[0:2], nbands + left_arm.shape[-1] + right_arm.shape[-1])
    )

    smoothed_spectrum[:, :, : left_arm.shape[-1]] = left_arm
    smoothed_spectrum[
        :, :, left_arm.shape[-1] : -right_arm.shape[-1]  # noqa
    ] = cube
    smoothed_spectrum[:, :, -right_arm.shape[-1] :] = right_arm  # noqa

    mu = convolve1d(smoothed_spectrum, window, axis=2) / window_size
    musq = convolve1d(smoothed_spectrum**2, window, axis=2) / window_size
    sigma = np.sqrt(musq - mu**2)

    mu = mu[:, :, window_size:-window_size]
    sigma = sigma[:, :, window_size:-window_size]

    return mu, sigma
