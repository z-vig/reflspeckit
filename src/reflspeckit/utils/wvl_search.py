# Dependencies
import numpy as np

# Top-Level Imports
from reflspeckit.data_classes import Wavelength, WvlUnit
from reflspeckit._errors import WavelengthUnitError


def find_wvl(
    wvls_input: Wavelength | np.ndarray, targetwvl: float, unit: WvlUnit = "nm"
):
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
    if isinstance(wvls_input, Wavelength):
        if wvls_input.unit != unit:
            raise WavelengthUnitError(
                f"{unit} does not match {wvls_input.unit}"
            )
        else:
            wvls = wvls_input.values
    else:
        wvls = wvls_input

    idx = np.argmin(np.abs(wvls - targetwvl))
    return idx, wvls[idx]
