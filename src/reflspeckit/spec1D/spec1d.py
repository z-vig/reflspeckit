# Built-Ins
from copy import copy

# Dependencies
import numpy as np
import numpy.typing as npt

# Top-Level Imports
from reflspeckit.data_classes import (
    Wavelength,
    WvlUnit,
    FilterMethodLiteral,
    ContinuumMethodLiteral,
)
from reflspeckit._errors import DimensionError

# Relative Imports
from .filtering import box_filter_single
from .outlier_detection import remove_outliers
from .continuum_removal import double_line
from .absorption_feature1d import AbsorptionFeature1D


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

        self.original_spectrum = copy(spec_arr)
        self.no_outliers: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.filtered: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.noise: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.contrem: npt.NDArray = np.full_like(spec_arr, np.nan)
        self.continuum: npt.NDArray = np.full_like(spec_arr, np.nan)

        self._validate()

    def outlier_removal(self, sigma_threshold: float = 1.5):
        no_outliers = remove_outliers(self.spectrum, sigma_threshold)
        self.no_outliers = no_outliers
        self.spectrum = no_outliers

    def noise_reduction(
        self,
        method: FilterMethodLiteral,
        filter_width: int,
    ):
        """
        Apply noise reduction to the spectrum.
        """
        if method == "box_filter":
            self.filtered, self.noise = box_filter_single(
                self.spectrum, filter_width
            )
            self.spectrum = self.filtered

    def continuum_removal(
        self,
        method: ContinuumMethodLiteral,
    ):
        if method == "double_line":
            self.contrem, self.continuum = double_line(
                self.spectrum, self.wavelength
            )
            self.spectrum = self.contrem

    def fit_absorption(
        self, low_wvl: float, high_wvl: float, unit: WvlUnit = "nm"
    ) -> AbsorptionFeature1D:
        if np.all(np.isnan(self.contrem)):
            raise ValueError("Continuum Removal has not been performed yet.")
        feature = AbsorptionFeature1D(
            self.contrem, self.wavelength, low_wvl, high_wvl, unit
        )
        return feature

    def _validate(self):
        if self.spectrum.ndim > 1:
            raise DimensionError(
                "Input spectrum array has too many dimensions "
                f"({self.spectrum.ndim})"
            )
