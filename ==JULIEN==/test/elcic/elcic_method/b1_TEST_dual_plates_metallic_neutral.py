import numpy as np
import espressomd
import numpy as np

from test.elcic.elcic_method.common.elcic_energy_accuracy_convergence import run as run_elcic_energy_accuracy_convergence
from test.elcic.elcic_method.common.elcic_dipole_shifting import run as run_elcic_dipole_shifting

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params = {
    "lx": 50.0,
    "ly": 50.0,
    "lz": 11.0,
    "gap_size": 7.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
    "charges": [+1, -1],
    'pw_error': 1e-6,
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 1])],
    "title": "Dual Plates, Both Metallic, Neutral"
}

params["positions"] = [np.array([1, 2, 0.5]), np.array([4, 5, 0.5])]
run_elcic_energy_accuracy_convergence(system, params)


# run_elcic_dipole_shifting(system, z_pos_count=1, params=params)
