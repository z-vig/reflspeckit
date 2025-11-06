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


class Interpolator:
    def __init__(self, xpts: np.ndarray, ypts: np.ndarray):
        """
        Parameters
        ----------
        xpts: np.ndarray
            1-D Array defining the tie points over which to interpolate.
        ypts: np.ndarray
            1-D Array representing the y points corresponding to the x points.
        """
        self.xpts = xpts
        self.ypts = ypts

    def linear(self, xvals: np.ndarray) -> np.ndarray:
        """
        Runs a linear interpolation over x values.
        """
        funclist: list[Callable] = []
        nseg = len(self.xpts) - 1
        for n in range(nseg):
            x1 = float(self.xpts[n])
            x2 = float(self.xpts[n + 1])
            y1 = self.ypts[n]
            y2 = self.ypts[n + 1]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            funclist.append(lambda x, m=m, b=b: m * x + b)

        funclist.insert(0, funclist[0])
        funclist.append(funclist[-1])

        condlist = [xvals < self.xpts[0]]
        for n in range(nseg):
            if n == nseg - 1:
                cond = (xvals >= self.xpts[n]) & (xvals <= self.xpts[n + 1])
            else:
                cond = (xvals >= self.xpts[n]) & (xvals < self.xpts[n + 1])
            condlist.append(cond)
        condlist.append(xvals > self.xpts[-1])

        result = np.full(xvals.size, np.nan)
        for cnd, fnc in zip(condlist, funclist):
            result[cnd] = fnc(xvals[cnd])
        return result
