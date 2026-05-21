import copy

import espressomd
import numpy as np

def get_energy(params:dict):
    """
    Example params:
    params = {
        "lx": 50.0,
        "ly": 50.0,
        "gap_size": 20.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+0.3, -0.6],
        "positions": [np.array([1, 2, 3]), np.array([4, 5, 6])],
        "pw_error": 1e-8,
    }
    params["lz"] = params["gap_size"] + 40
    
    """
    return 0.0
