# Dependencies
import numpy as np
import numpy.typing as npt


class CubeFitResult:
    def __init__(
        self,
        xdata: npt.NDArray,
        ydata: npt.NDArray,
        model_domain: npt.NDArray,
        model: npt.NDArray,
        beta: npt.NDArray,
        res: npt.NDArray,
        order: int,
    ):
        self.xdata = xdata
        self.ydata = ydata
        self.model_domain = model_domain
        self.model = model
        self.beta = beta
        self.res = res
        self._order = order

    def r_squared(self) -> npt.NDArray[np.float32]:
        """
        Returns the coefficient of determination (R<sup>2</sup>)
        """
        ss_res = np.sum(self.res**2, axis=2)
        ss_tot = np.sum((self.ydata - np.mean(self.ydata, axis=-1)[:, :, None]) ** 2)
        return 1 - (ss_res / ss_tot)

    def eval(self, x: npt.NDArray):
        """
        Returns dependent variable value given independent variable.
        """
        xvals = np.vander(x, self._order + 1)
        return np.einsum("ij,...j->...i", xvals, self.beta)


def polyfit_cube(
    spectral_cube: np.ndarray,
    wvl: np.ndarray,
    order: int,
    model_xvals: np.ndarray | None = None,
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
    X_fit = np.vander(wvl, order + 1)
    prefix = np.linalg.inv(X_fit.T @ X_fit) @ X_fit.T
    beta = np.einsum("ij,...j->...i", prefix, spectral_cube)

    if model_xvals is None:
        X_test = X_fit
        comparison_idx = np.arange(0, spectral_cube.shape[2])
        model_domain = wvl
    else:
        X_test = np.vander(model_xvals, order + 1)
        comparison_idx_list = []
        for i in model_xvals:
            comparison_idx_list.append(np.argmin(np.abs(wvl - i)))
        comparison_idx = np.array(comparison_idx_list)
        model_domain = model_xvals

    model = np.einsum("ij,...j->...i", X_test, beta)
    residual = spectral_cube[:, :, comparison_idx] - model

    return CubeFitResult(wvl, spectral_cube, model_domain, model, beta, residual, order)
