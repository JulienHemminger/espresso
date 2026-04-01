import numpy as np
import espressomd
import numpy as np

from test.elcic.elcic_method.common.elcic_energy_accuracy_convergence import run as run_elcic_energy_accuracy_convergence
from test.elcic.elcic_method.common.elcic_dipole_shifting import run as run_elcic_dipole_shifting

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params = {
        "lx": 9.0,
        "ly": 12.0,
        "lz": 19.0,
        "gap_size": 15.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1, # metallic: Δ = -1
        "charges": [-1.7, +0.4],
        'pw_error': 1e-8,
        "positions": [np.array([2, 5, 0]), np.array([8, 3, 0])],
        "title": "Single Plate, Metallic, Non-Neutral",
    }

params["positions"] = [np.array([2, 5, 0.01]), np.array([8, 3, 0.02])]
run_elcic_energy_accuracy_convergence(system, params)

run_elcic_dipole_shifting(system, z_pos_count=32, params=params)


"""
Für "Single Plate, Metallic, Non-Neutral" ...


Auch hier gabs bei Espressos "legacy" ELC Methode den Fehler "RuntimeError: ELC does not work for non-neutral systems and non-metallic dielectric contrast." gibt. Auch wenn die eine Platte metallisch ist. Aber ich denke das liegt daran dass die andere Plate "weg" ist (delta_top=0)
"""
