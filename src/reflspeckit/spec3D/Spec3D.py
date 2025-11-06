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
)
from reflspeckit._errors import DimensionError

# Relative Imports
from .outlier_detection import remove_outliers
from .filtering import box_filter_cube


class ProcessingFlag(Enum):
    RAW = auto()
    OUTLIERS_REMOVED = auto()
    FILTERED = auto()


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
        self.wvl = Wavelength(wvl_arr, "nm")

        self._processing_flag = ProcessingFlag.RAW

        self._validate()

    def outlier_removal(self, sigma_threshold: float = 1.5):
        self.cube = remove_outliers(self.cube, sigma_threshold)
        self._processing_flag = ProcessingFlag.OUTLIERS_REMOVED

    def noise_reduction(
        self, method: FilterMethod | FilterMethodLiteral, filter_width: int
    ):
        if self._processing_flag == ProcessingFlag.FILTERED:
            return
        if self._processing_flag != ProcessingFlag.OUTLIERS_REMOVED:
            self.outlier_removal()
        if method == "box_filter":
            self.cube, _ = box_filter_cube(self.cube, filter_width)

    def _validate(self):
        if self.cube.ndim > 3:
            raise DimensionError(
                "Input spectrum array has too many dimensions "
                f"({self.cube.ndim})"
            )
