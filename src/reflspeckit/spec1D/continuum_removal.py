# Dependencies
import numpy as np

# Relative Imports
from .utils import Interpolator

# Top-Level Imports
from reflspeckit.data_classes import Wavelength


def double_line(
    spectrum: np.ndarray, wvls: Wavelength
) -> tuple[np.ndarray, np.ndarray]:
    """
    Double-line continuum removal.

    Parameters
    ----------
    spectrum: np.ndarray
        Filtered spectrum with the continuum in place.
    wvls: Wavelength
        Wavelength values in the form of a Wavelength object.

    Returns
    -------
    continuum_removed: np.ndarray
        Input spectrum with the continuum removed.
    continuum: np.ndarray
        Continuum values for the spectrum.
    """
    # Getting initial continuum line parameters
    anchor_pts = np.array([700, 1550, 2600])
    if wvls.unit == "nm":
        pass
    if wvls.unit == "um":
        anchor_pts = anchor_pts * 10**-3
    if wvls.unit == "m":
        anchor_pts = anchor_pts * 10**-9

    cont1_band_idx = np.empty(anchor_pts.size, dtype=np.int16)
    cont1_band_wvls = np.empty(anchor_pts.size, dtype=np.float64)
    for n in range(anchor_pts.size):
        idx = np.argmin(np.abs(wvls - anchor_pts[n]))
        cont1_band_idx[n] = idx
        cont1_band_wvls[n] = wvls.values[idx]

    cont1_spectrum_values = spectrum[cont1_band_idx]

    CI1 = Interpolator(cont1_band_wvls, cont1_spectrum_values)
    continuum1 = CI1.linear(wvls.values)

    continuum1_removed = spectrum / continuum1

    cont2_ranges: dict[str, tuple[float, float]] = {
        "range1": (650.0, 1000.0),
        "range2": (1250.0, 1600.0),
        "range3": (2000.0, 2800.0),
    }

    def scale_dict_values(
        data: dict[str, tuple[float, float]], factor: float
    ) -> dict[str, tuple[float, float]]:
        return {k: (v[0] * factor, v[1] * factor) for k, v in data.items()}

    if wvls.unit == "nm":
        pass
    elif wvls.unit == "um":
        cont2_ranges = scale_dict_values(cont2_ranges, 10**-3)
    elif wvls.unit == "m":
        cont2_ranges = scale_dict_values(cont2_ranges, 10**-9)
    else:
        raise ValueError("Wavelength Units not valid.")

    cont2_band_idx = np.empty(len(cont2_ranges), dtype=np.int16)

    # Locating dynamic maxima for second continuum parameters
    for n, val in enumerate(cont2_ranges.values()):
        lo_idx = np.argmin(np.abs(wvls.values - val[0]))
        hi_idx = np.argmin(np.abs(wvls.values - val[1]))

        max_spectrum_idx = (
            np.argmax(
                continuum1_removed[np.arange(lo_idx, hi_idx, dtype=np.int16)]
            )
            + lo_idx
        )
        cont2_band_idx[n] = max_spectrum_idx

    cont2_band_wvls = wvls.values[cont2_band_idx]
    cont2_spectrum_values = spectrum[cont2_band_idx]

    CI2 = Interpolator(cont2_band_wvls, cont2_spectrum_values)
    continuum2 = CI2.linear(wvls.values)
    continuum2_removed = spectrum / continuum2

    return continuum2_removed, continuum2
