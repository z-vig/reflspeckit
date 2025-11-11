# Dependencies
import numpy as np
import numpy.typing as npt

# Relative Imports
from .polyfit import polyfit_single

# Top-Level Imports
from reflspeckit.data_classes import Wavelength, WvlUnit
from reflspeckit.utils import find_wvl
from reflspeckit._errors import WavelengthUnitError


class AbsorptionFeature1D:
    """
    Stores band parameter values for a specific absorption feature within a
    spectrum.

    Parameters
    ----------
    spectrum: npt.NDArray
        1-dimensional Array of spectrum values.
    low_wavelength: float
        Low end of absorption feature search.
    high_wavelength: float
        High end of absorption feature search.

    Attributes
    ----------

    Notes
    -----
    """

    def __init__(
        self,
        spectrum: npt.NDArray,
        wavelength: Wavelength,
        low_wavelength: float,
        high_wavelength: float,
        unit: WvlUnit = "nm",
        fit_order: int = 2,
    ) -> None:
        self.spec = spectrum
        self.wvl = wavelength
        self.lowvl = low_wavelength
        self.hiwvl = high_wavelength

        if unit != self.wvl.unit:
            raise WavelengthUnitError(
                f"Absorption fit wavelengths are in units of {unit} whereas "
                f"the provided wavelegnth array is in units of {self.wvl.unit}"
            )

        low_idx, _lowwvl_exact = find_wvl(self.wvl.values, low_wavelength)
        high_idx, _highwvl_exact = find_wvl(self.wvl.values, high_wavelength)

        fit_slice = slice(low_idx, high_idx)

        self.fit_wvl = self.wvl.values[fit_slice]

        self.fit_result = polyfit_single(
            self.spec[fit_slice], self.wvl.values[fit_slice], fit_order
        )

    def calculate_center(self) -> tuple[float, float]:
        center_idx = np.argmin(self.fit_result.model)
        if (center_idx == 0) or (center_idx == self.spec.size - 1):
            return np.nan, np.nan
        else:
            return self.fit_result.ydata[center_idx], self.fit_wvl[center_idx]
