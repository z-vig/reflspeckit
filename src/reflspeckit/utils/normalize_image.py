# utils/norm_image_controlled.py

# Standard Libraries
from copy import copy
from typing import Optional

# External Libraries
import numpy as np


def normalize_image(
    image_data: np.ndarray,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    min_val: float = 0,
    max_val: float = 1,
):
    """
    Normalizes an image from `min_val` to `max_val`, while cutting off image
    values below `low_threshold` and above `high_threshold` so that any value
    <`low`=`min_val` and any value >`high`=`max_val`.
    """
    norm_img = copy(image_data)
    no_nans = norm_img[np.isfinite(norm_img)]

    if (low_threshold is None) or (high_threshold is None):
        low = no_nans.min()
        high = no_nans.max()
    else:
        low = low_threshold
        high = high_threshold

    norm_img[norm_img > high] = high
    norm_img[norm_img < low] = low
    norm_img = (norm_img - low) / (high - low)
    norm_img = min_val + (norm_img * (max_val - min_val))
    return norm_img
