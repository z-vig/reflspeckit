# Dependencies
import numpy as np

# Top-Level Imports
from reflspeckit.data_classes import (
    Wavelength,
    WvlUnit,
    FilterMethod,
    FilterMethodLiteral,
)
from reflspeckit._errors import DimensionError

# Relative Imports
from .filtering import box_filter_single
from .outlier_detection import remove_outliers


class Spec1D:
    """
    Main class for handling 1-dimensional spectral data (i.e. a single
    spectrum).

    Parameters
    ----------
    spec_arr: np.ndarray
    wvl_arr: np.ndarray
        Wavelength values of the spectrum. Must be a 1D array.
    unit: WvlUnit, optional
        Unit for the wavelength values. Default is "nm", but options include:

        - "nm": Nanometers
        - "um": Microns (micrometers)
        - "m": Meters

    Notes
    -----
    See Spec3D for an equivalent class that handles 3-dimensional spectral data
    image cubes.
    """

    def __init__(
        self, spec_arr: np.ndarray, wvl_arr: np.ndarray, unit: WvlUnit = "nm"
    ):
        self.spectrum = spec_arr
        self.wavelength = Wavelength(wvl_arr, "nm")

        self.no_outliers: np.ndarray = np.full_like(spec_arr, np.nan)
        self.filtered: np.ndarray = np.full_like(spec_arr, np.nan)
        self.noise: np.ndarray = np.full_like(spec_arr, np.nan)

        self._validate()

    def outlier_removal(self, sigma_threshold: float = 1.5):
        self.no_outliers = remove_outliers(self.spectrum, sigma_threshold)

    def noise_reduction(
        self,
        method: FilterMethod | FilterMethodLiteral,
        filter_width: int,
        remove_outliers: bool = True,
        sigma_threshold: float = 1.5,
    ):
        """
        Apply noise reduction to the spectrum.
        """
        if remove_outliers:
            self.outlier_removal(sigma_threshold)
            spectrum_to_use = self.no_outliers
        else:
            spectrum_to_use = self.spectrum
        if method == "box_filter":
            self.filtered, self.noise = box_filter_single(
                spectrum_to_use, filter_width
            )

    def _validate(self):
        if self.spectrum.ndim > 1:
            raise DimensionError(
                "Input spectrum array has too many dimensions "
                f"({self.spectrum.ndim})"
            )
