import numpy as np
import espressomd
import numpy as np

from test.elcic.elcic_method.common.elcic_energy_accuracy_convergence import run as run_elcic_energy_accuracy_convergence
from test.elcic.elcic_method.common.elcic_dipole_shifting import run as run_elcic_dipole_shifting

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params = {
    "lx": 100.0,
    "ly": 100.0,
    "lz": 20.0,
    "gap_size": 15.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
    "charges": [+1, -1],
    'pw_error': 1e-8,
    "positions": [np.array([2, 5, 3]), np.array([8, 3, 1])],
    "title": "Dual Plates, Both Metallic, Neutral"
}# ANA: Iteration 10: energy(k_val=1024, n_val=100) = -0.35079175141122165

#params["positions"] = [np.array([2, 5, 0.01]), np.array([8, 3, 0.02])]
#run_elcic_energy_accuracy_convergence(system, params)


run_elcic_dipole_shifting(system, z_pos_count=4, params=params)

