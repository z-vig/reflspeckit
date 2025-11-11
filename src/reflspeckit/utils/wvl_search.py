# Dependencies
import numpy as np


def find_wvl(wvls: np.ndarray, targetwvl: float):
    """
        findλ(λ.targetλ)

    Given a list of wavelengths, `wvls`, find the index of a `targetwvl` and
    the actual wavelength closest to your target.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength array to search in.
    targetwvl:
        Wavelength to search for.

    Returns
    -------
    idx: int
        Index of the found wavelength.
    wvl: float
        Actual wavelength that is closest to the target wavelength (at idx).
    """

    idx = np.argmin(np.abs(wvls - targetwvl))
    return idx, wvls[idx]
