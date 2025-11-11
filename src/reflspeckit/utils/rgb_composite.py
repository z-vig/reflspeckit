# utils/rgb_composite.py

# External Imports
import numpy as np

# Local Imports
from .normalize_image import normalize_image


def make_rgb_composite(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    low_percentile: int = 5,
    high_percentile: int = 95,
):
    """
    Creates an RGB false color composite image from three image-like arrays.
    """
    rdata = r[np.isfinite(r)]
    gdata = g[np.isfinite(g)]
    bdata = b[np.isfinite(b)]

    rnorm = normalize_image(
        r,
        float(np.percentile(rdata, low_percentile)),
        float(np.percentile(rdata, high_percentile)),
    )
    gnorm = normalize_image(
        g,
        float(np.percentile(gdata, low_percentile)),
        float(np.percentile(gdata, high_percentile)),
    )
    bnorm = normalize_image(
        b,
        float(np.percentile(bdata, low_percentile)),
        float(np.percentile(bdata, high_percentile)),
    )

    rgb_composite = np.concat(
        [
            rnorm[:, :, np.newaxis],
            gnorm[:, :, np.newaxis],
            bnorm[:, :, np.newaxis],
        ],
        axis=2,
    )

    return rgb_composite
