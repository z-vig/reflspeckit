# Dependencies
import numpy as np
import numpy.typing as npt

# Top-Level Imports
from reflspeckit.data_classes import (
    Wavelength,
    WvlUnit,
    FilterMethod,
    FilterMethodLiteral,
    ContinuumMethod,
    ContinuumMethodLiteral,
)
from reflspeckit._errors import DimensionError

# Relative Imports
from .filtering import box_filter_single
from .outlier_detection import remove_outliers
from .continuum_removal import double_line


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
        self, spec_arr: npt.NDArray, wvl_arr: npt.NDArray, unit: WvlUnit = "nm"
    ):
        self.spectrum = spec_arr
        self.wavelength = Wavelength(wvl_arr, "nm")

        self.no_outliers: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.filtered: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.noise: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.contrem: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.continuum: npt.NDArray = np.full_like(spec_arr, np.nan)

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

    def continuum_removal(
        self,
        method: ContinuumMethod | ContinuumMethodLiteral,
        filter_width: int = 5,
    ):
        if np.all(np.isnan(self.filtered)):
            self.noise_reduction("box_filter", filter_width)

        if method == "double_line":
            self.contrem, self.continuum = double_line(
                self.filtered, self.wavelength
            )

    def _validate(self):
        if self.spectrum.ndim > 1:
            raise DimensionError(
                "Input spectrum array has too many dimensions "
                f"({self.spectrum.ndim})"
            )
