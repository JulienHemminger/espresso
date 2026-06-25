
import os
import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Callable, Any

def lerp(a: Any, b: Any, t: float) -> Any:
    """Linear interpolation between a and b. Handles scalars and numpy arrays."""
    return (1 - t) * a + t * b

def lerp_dict(start_params: dict, end_params: dict, t: float) -> dict:
    """Recursively interpolates matching keys between two dictionaries."""
    lerp_params = {}
    for key in start_params:
        v1, v2 = start_params[key], end_params[key]
        if isinstance(v1, list):
            # Handles lists
            lerp_params[key] = [lerp(np.array(p1), np.array(p2), t) for p1, p2 in zip(v1, v2)]
        else:
            lerp_params[key] = lerp(v1, v2, t)
    return lerp_params