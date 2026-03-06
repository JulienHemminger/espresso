import numpy as np


def has_downward_trend(data):
    """
    Returns True if the linear regression slope is negative.
    """
    x = np.arange(len(data))
    y = np.array(data)
    slope, _ = np.polyfit(x, y, 1)
    return slope < 0
