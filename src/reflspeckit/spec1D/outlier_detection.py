# Dependencies
import numpy as np

# Relative Imports
from .utils import round_to_odd
from .filtering import box_filter_single


def remove_outliers(original_spectrum: np.ndarray, threshold: float):
    """
    Detects and replaces statistical outliers in a 1D spectrum using
    neighbor-based interpolation.

    An outlier is defined as any value that deviates from a local moving
    average by more than a specified number of standard deviations
    (default is 2). Identified outliers are replaced by the mean of
    their immediate neighbors (i.e., the values before and after),
    excluding the outlier itself. Edge cases are handled by treating
    missing neighbors as NaN.

    Parameters
    ----------
    original_spectrum : np.ndarray
        The 1D input spectrum array to process. Must be a numeric NumPy array.
    threshold : float, optional
        The Z-score threshold to use for detecting outliers. Any value with a
        Z-score greater than `threshold` (in absolute value) is considered
        an outlier. Default is 2.

    Returns
    -------
    np.ndarray
        A new spectrum array with outliers replaced by the mean of their
        immediate neighbors. The original input is not modified.

    Notes
    -----
    - The local mean and standard deviation are computed using a moving
      window that spans 10% of the spectrum length (minimum size of 3
      and always rounded to an odd number).
    - To avoid circular wraparound, the first and last elements only
      use their single available neighbor for replacement.
    - This method is robust to isolated spikes, but may not perform well
      for broad or clustered outliers.

    Examples
    --------
    >>> import numpy as np
    >>> spectrum = np.array([1.0, 1.1, 1.2, 10.0, 1.3, 1.2, 1.1])
    >>> outlier_removal(spectrum)
    array([1. , 1.1, 1.2, 1.25, 1.3, 1.2, 1.1])
    """
    spec = np.copy(original_spectrum)
    window_size = max(round_to_odd(spec.size * 0.1), 3)
    mu, sig = box_filter_single(spec, window_size=window_size)
    zscore = (spec - mu) / sig
    outlier_idx = abs(zscore) > threshold

    neighbors = np.stack([np.roll(spec, -1), np.roll(spec, 1)], axis=1)

    neighbors[0, 1] = np.nan
    neighbors[-1, 0] = np.nan

    replacement = np.nanmean(neighbors, axis=1)

    spec[outlier_idx] = replacement[outlier_idx]

    return spec
