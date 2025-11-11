# Standard Libraries
from enum import Enum, auto

# Dependencies
import numpy as np

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
from reflspeckit.utils import find_wvl, make_rgb_composite

# Relative Imports
from .outlier_detection import remove_outliers
from .filtering import box_filter_cube
from .continuum_removal import double_line
from .absorption_feature3d import AbsorptionFeature3D


class ProcessingFlag(Enum):
    RAW = auto()
    OUTLIERS_REMOVED = auto()
    FILTERED = auto()
    CONTINUUM_REMOVED = auto()


class Spec3D:
    """
    Main class for handling 3-dimensional spectral image cubes (i.e. an image
    where each pixel corresponds to a full spectrum).

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

    Image cubes must be smaller than memory to use the standard Spec3D class.
    For image cubes larger than memory, see `StreamingSpec3D`.
    """

    def __init__(
        self, spec_arr: np.ndarray, wvl_arr: np.ndarray, unit: WvlUnit = "nm"
    ):
        self.cube = spec_arr
        self.wavelength = Wavelength(wvl_arr, "nm")
        self._albedo_idx, _ = find_wvl(self.wavelength, 1580, unit="nm")

        self._processing_flag = ProcessingFlag.RAW

        self._validate()

    def outlier_removal(self, sigma_threshold: float = 1.5):
        print("Removing Outliers...")
        self.cube = remove_outliers(self.cube, sigma_threshold)
        self._processing_flag = ProcessingFlag.OUTLIERS_REMOVED

    def noise_reduction(
        self, method: FilterMethod | FilterMethodLiteral, filter_width: int
    ):
        print("Running Noise Reduction...")
        if self._processing_flag == ProcessingFlag.FILTERED:
            return
        if self._processing_flag != ProcessingFlag.OUTLIERS_REMOVED:
            self.outlier_removal()
        if method == "box_filter":
            self.cube, _ = box_filter_cube(self.cube, filter_width)
            self.albedo_image = self.cube[:, :, self._albedo_idx]

        self._processing_flag = ProcessingFlag.FILTERED

    def continuum_removal(
        self, method: ContinuumMethod | ContinuumMethodLiteral
    ):
        print("Removing Spectral Continuum...")
        if self._processing_flag == ProcessingFlag.CONTINUUM_REMOVED:
            return
        if self._processing_flag != ProcessingFlag.FILTERED:
            self.noise_reduction("box_filter", 7)
        if method == "double_line":
            self.cube, _ = double_line(self.cube, self.wavelength)

        self._processing_flag = ProcessingFlag.CONTINUUM_REMOVED

    def fit_absorption(
        self, low_wvl: float, high_wvl: float, unit: WvlUnit = "nm"
    ) -> AbsorptionFeature3D:
        print(
            f"Fitting Absorption Feature from {low_wvl} to {high_wvl} "
            f"{self.wavelength.unit}"
        )
        if self._processing_flag != ProcessingFlag.CONTINUUM_REMOVED:
            raise ValueError("Continuum Removal has not been performed yet.")
        feature = AbsorptionFeature3D(
            self.cube, self.wavelength, low_wvl, high_wvl, unit
        )
        return feature

    def make_m3_rgb(self):
        if self._processing_flag != ProcessingFlag.CONTINUUM_REMOVED:
            raise ValueError("Continuum Removal has not been performed yet.")
        bnd1 = self.fit_absorption(0.789, 1.309, unit="um")
        bnd2 = self.fit_absorption(1.658, 2.498, unit="um")

        ibd1 = bnd1.calculate_ibd()
        ibd2 = bnd2.calculate_ibd()

        rgb = make_rgb_composite(ibd1, ibd2, self.albedo_image)

        return rgb

    def _validate(self):
        if self.cube.ndim < 3:
            raise DimensionError(
                "Input spectrum array does not have enough dimensions "
                f"({self.cube.ndim})"
            )
        if self.cube.ndim > 3:
            raise DimensionError(
                "Input spectrum array has too many dimensions "
                f"({self.cube.ndim})"
            )
