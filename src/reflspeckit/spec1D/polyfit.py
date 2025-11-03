# Standard Libraries
from typing import Optional

# Dependencies
import numpy as np
import numpy.typing as npt


class SingleFitResult:
    def __init__(
        self,
        xdata: npt.NDArray,
        ydata: npt.NDArray,
        model: npt.NDArray,
        beta: npt.NDArray,
        res: npt.NDArray,
    ):
        self.xdata = xdata
        self.ydata = ydata
        self.model = model
        self.beta = beta
        self.res = res

    def r_squared(self) -> npt.NDArray[np.float32]:
        """
        Returns the coefficient of determination (R<sup>2</sup>)
        """
        ss_res = np.sum(self.res**2)
        ss_tot = np.sum((self.ydata - np.mean(self.ydata)) ** 2)
        return 1 - (ss_res / ss_tot)

    def eval(self, x: npt.NDArray):
        """
        Returns dependent variable value given independent variable.
        """
        X = np.vander(x, self.beta.size, increasing=True)
        return (X @ self.beta[:, None])[:, 0]


def polyfit_single(
    spectrum: np.ndarray,
    wvl: np.ndarray,
    order: int = 1,
    design_matrices: Optional[tuple[np.ndarray, ...]] = None,
    return_coefficients: bool = False,
) -> SingleFitResult:
    """
    Fits a polynomial of order `N` to `x` and `y` data. Optionally can be used
    in a loop by specifying the `design_matrix` elements.

    Parameters
    ----------
    spectrum: np.ndarray
        Y Data. Spectral Data.
    wvl: np.ndarray
        X Data. Wavelengths. Can be None if `design_matrices` are supplied.
    order: int
        Order of polynomial fit. Can be None if `design_matrices` are supplied.
    design_matrices: Nothing or tuple[np.ndarray]
        Three design matrix components. X, Xt and XtX. If None (default),
        these are calculated from `wvl` data.
    return_coefficients: bool, optional
        If True (defualt), returns fit coefficients rather than a fit line.

    Returns
    -------
    beta: np.ndarray
        Coefficients of the fit.
    """

    if design_matrices is None:
        X = np.vander(wvl, order + 1, increasing=True)
        Xt = X.T
        XtX = Xt @ X
    else:
        X, Xt, XtX = design_matrices

    beta = np.linalg.inv(XtX) @ (Xt @ spectrum)
    model = X @ beta
    residual = model - spectrum

    return SingleFitResult(wvl, spectrum, model, beta, residual)
