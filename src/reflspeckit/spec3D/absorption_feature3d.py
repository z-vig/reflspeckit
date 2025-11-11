# Dependencies
import numpy as np
import numpy.typing as npt

# Relative Imports
from .polyfit import polyfit_cube

# Top-Level Imports
from reflspeckit.data_classes import Wavelength, WvlUnit
from reflspeckit.utils import find_wvl
from reflspeckit._errors import WavelengthUnitError


class AbsorptionFeature3D:
    def __init__(
        self,
        contrem_spectrum: npt.NDArray,
        wavelength: Wavelength,
        low_wavelength: float,
        high_wavelength: float,
        unit: WvlUnit = "nm",
        fit_order: int = 4,
    ):
        self.spec = contrem_spectrum
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
        self._fit_spec = self.spec[:, :, fit_slice]

        self.fit_result = polyfit_cube(
            self.spec[:, :, fit_slice], self.wvl.values[fit_slice], fit_order
        )

    def calculate_center(self) -> npt.NDArray:
        center_idx_arr = np.argmin(self.fit_result.model, axis=2)
        bc_map = self.fit_wvl[center_idx_arr]
        bc_map[center_idx_arr == 0] = np.nan
        bc_map[center_idx_arr == self.spec.shape[-1] - 1] = np.nan

        return bc_map

    def calculate_ibd(self) -> npt.NDArray:
        return np.sum(1 - self._fit_spec, axis=2)
