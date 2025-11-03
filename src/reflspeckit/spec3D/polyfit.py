# Dependencies
import numpy as np
import numpy.typing as npt


class CubeFitResult:
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
        ss_res = np.sum(self.res**2, axis=2)
        ss_tot = np.sum(
            (self.ydata - np.mean(self.ydata, axis=-1)[:, :, None]) ** 2
        )
        return 1 - (ss_res / ss_tot)

    def eval(self, x: float | npt.NDArray):
        """
        Returns dependent variable value given independent variable.
        """
        return self.beta @ x


def polyfit_cube(
    spectral_cube: np.ndarray, wvl: np.ndarray, order: int
) -> CubeFitResult:
    """
    Performs polynomial fits for an entire spectral cube of data.

    Parameters
    ----------
    spectral_cube: np.ndarray
        Spectral data cube.
    wvl: np.ndarray
        Wavelength values.
    order: int
        Order of polyfit.
    return_coefficients: bool, optional
        If True (defualt), returns a cube of coefficient rather than fit lines.

    Returns
    -------
    fit_cube: np.ndarray
        Either a cube of fitted coefficients or fitted lines.
    """
    X = np.vander(wvl, order + 1)
    prefix = np.linalg.inv(X.T @ X) @ X.T
    beta = np.einsum("ij,...j->...i", prefix, spectral_cube)
    model = np.einsum("ij,...j->...i", X, beta)
    residual = spectral_cube - model

    return CubeFitResult(wvl, spectral_cube, model, beta, residual)
