# Standard Libraries
from typing import Literal, Optional
from enum import StrEnum

# Dependencies
import numpy as np

# Top-Level Imports
from reflspeckit._errors import DimensionError

# Wavelength unit type alias
WvlUnit = Literal["nm", "um", "m"]


class Wavelength:
    def __init__(
        self,
        values: np.ndarray,
        unit: WvlUnit,
        resolution: Optional[float] = None,
    ):
        self.values = values
        self.unit = unit
        self.resolution = resolution
        self._validate()

    def __array__(self):
        return self.values

    def to_nm(self):
        if self.unit == "nm":
            pass
        elif self.unit == "um":
            self.values *= 10**3
        elif self.unit == "m":
            self.values *= 10**9

        self.unit = "nm"

    def to_um(self):
        if self.unit == "nm":
            self.values *= 10**-3
        elif self.unit == "um":
            pass
        elif self.unit == "m":
            self.values *= 10**6

        self.unit = "um"

    def to_m(self):
        if self.unit == "nm":
            self.values *= 10**-9
        elif self.unit == "um":
            self.values *= 10**-6
        elif self.unit == "m":
            pass

        self.unit = "m"

    def _validate(self):
        if self.values.ndim > 1:
            raise DimensionError(
                "Wavelength array has too many dimensions "
                f"({self.values.ndim})"
            )


class FilterMethod(StrEnum):
    BOX_FILTER = "box_filter"


type FilterMethodLiteral = Literal["box_filter"]


class ContinuumMethod(StrEnum):
    DOUBLE_LINE = "double_line"


type ContinuumMethodLiteral = Literal["double_line"]
