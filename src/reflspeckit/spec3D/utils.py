# Standard Libraries
from typing import Callable

# Dependencies
import numpy as np


def round_to_odd(num):
    """
    Rounds a number to the nearest odd integer.
    """
    r = round(num, 0)
    if r % 2 == 0:
        if (num - r) != 0:
            return int(r + ((num - r) / abs(num - r)))
        else:
            return int(r - 1)
    else:
        return int(r)


class CubeInterpolator:
    def __init__(self, xpts: np.ndarray, ypts: np.ndarray):
        """
        Parameters
        ----------
        xpts: np.ndarray
            1-D Array defining the shared x points over which to interpolate
            the cube.
        ypts: np.ndarray
            3-D Array representing where the last axis contains the y points
            corresponding to the x points.
        """
        self.xpts = xpts
        self.ypts = ypts

    def linear(self, xvals: np.ndarray) -> np.ndarray:
        """
        Runs a linear interpolation over x values.
        """
        if self.xpts.ndim == 1:
            return self._linear_constX(xvals)
        elif self.xpts.ndim == 3:
            return self._linear_varX(xvals)
        else:
            raise ValueError(f"xvals is not valid. ({xvals.ndim} dims)")

    def _linear_constX(self, xvals: np.ndarray) -> np.ndarray:
        """
        Linear interpolation assuming that all pixels in a cube have the same
        tie points on the X axis. This method is called via `linear`.
        """
        # Build per-segment linear functions. Bind m and b into the lambda
        # default arguments to avoid late-binding closures.
        funclist: list[Callable] = []
        nseg = len(self.xpts) - 1
        for n in range(nseg):
            x1 = float(self.xpts[n])
            x2 = float(self.xpts[n + 1])
            y1 = self.ypts[:, :, n]
            y2 = self.ypts[:, :, n + 1]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # bind m and b so each function keeps its own slope/intercept
            funclist.append(
                lambda x, m=m, b=b: m[:, :, None] * x[None, None, :]
                + b[:, :, None]
            )

        funclist.insert(0, funclist[0])
        funclist.append(funclist[-1])

        # Create boolean masks per segment. Use left-inclusive, right-exclusive
        # bins except for the last segment which includes the right edge.
        condlist = [xvals < self.xpts[0]]
        for n in range(nseg):
            if n == nseg - 1:
                cond = (xvals >= self.xpts[n]) & (xvals <= self.xpts[n + 1])
            else:
                cond = (xvals >= self.xpts[n]) & (xvals < self.xpts[n + 1])
            condlist.append(cond)
        condlist.append(xvals > self.xpts[-1])

        result = np.full((*self.ypts.shape[:2], len(xvals)), np.nan)
        for cnd, fnc in zip(condlist, funclist):
            result[:, :, cnd] = fnc(xvals[cnd])
        return result

    def _linear_varX(self, xvals: np.ndarray) -> np.ndarray:
        nseg = self.xpts.shape[-1]
        slope_arr = np.empty((*self.ypts.shape[:2], self.xpts.shape[-1] - 1))
        intcp_arr = np.empty_like(slope_arr)
        for n in range(nseg - 1):
            x1 = self.xpts[:, :, n]
            x2 = self.xpts[:, :, n + 1]
            y1 = self.ypts[:, :, n]
            y2 = self.ypts[:, :, n + 1]
            slope_arr[:, :, n] = (y2 - y1) / (x2 - x1)
            intcp_arr[:, :, n] = y1 - slope_arr[:, :, n] * x1

        slope_arr = np.concat(
            [slope_arr, slope_arr[:, :, -1][:, :, None]], axis=2
        )
        intcp_arr = np.concat(
            [intcp_arr, intcp_arr[:, :, -1][:, :, None]], axis=2
        )

        eqn_arr = np.full((*self.ypts.shape[:2], xvals.size, 2), np.nan)
        for n in range(xvals.size):
            slope_arr_temp = slope_arr[:, :, 0]
            intcp_arr_temp = intcp_arr[:, :, 0]
            for j in range(self.xpts.shape[-1]):
                cond_arr_temp = xvals[n] > self.xpts[:, :, j]
                slope_arr_temp[cond_arr_temp] = slope_arr[cond_arr_temp, j]
                intcp_arr_temp[cond_arr_temp] = intcp_arr[cond_arr_temp, j]
            eqn_arr[:, :, n, 0] = slope_arr_temp
            eqn_arr[:, :, n, 1] = intcp_arr_temp

        line_arr = np.full((*self.ypts.shape[:2], xvals.size), np.nan)
        for n in range(xvals.size):
            line_arr[:, :, n] = (
                eqn_arr[:, :, n, 0] * xvals[n] + eqn_arr[:, :, n, 1]
            )
        return line_arr
