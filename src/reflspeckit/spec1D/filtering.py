# Dependencies
import numpy as np
import numpy.typing as npt
from scipy.signal import convolve

# Relative Imports
from .polyfit import polyfit_single


def box_filter_single(
    spectrum: np.ndarray, window_size: int = 5
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Smoothes a spectrum using a moving average.

    Parameters
    ----------
    original_spectrum: np.ndarray
        Non-smooth spectrum.
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
    spec: npt.NDArray = spectrum

    window = np.ones(window_size)
    nbands = spec.size

    edge_length = int(np.maximum(round(spec.size * 0.1, 0), 2))
    left_idx: npt.NDArray[np.int32] = np.arange(
        0, 1 + edge_length, dtype=np.int32
    )
    right_idx: npt.NDArray[np.int32] = np.arange(
        nbands - edge_length, nbands, dtype=np.int32
    )

    left_fit = polyfit_single(spec[left_idx], left_idx, 1)
    left_ext = np.arange(-window_size, 0)
    left_arm = left_fit.eval(left_ext)

    right_fit = polyfit_single(spec[right_idx], right_idx, 1)
    right_ext = np.arange(nbands, nbands + window_size)
    right_arm = right_fit.eval(right_ext)

    smoothed_spectrum = np.empty(nbands + left_arm.size + right_arm.size)

    smoothed_spectrum[: left_arm.size] = left_arm
    smoothed_spectrum[left_arm.size : -right_arm.size] = spec  # noqa
    smoothed_spectrum[-right_arm.size :] = right_arm  # noqa

    mu = convolve(smoothed_spectrum, window, mode="same") / window_size
    musq = convolve(smoothed_spectrum**2, window, mode="same") / window_size
    sigma = np.sqrt(musq - mu**2)

    mu = mu[window_size:-window_size]
    sigma = sigma[window_size:-window_size]

    return mu, sigma
