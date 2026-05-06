import numpy as np
import espressomd
from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence
import random

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params_count = 10
params_sets = []


for _ in range(params_count):
    params = {
        "lx": np.random.uniform(10.0, 20.0),
        "ly": np.random.uniform(10.0, 20.0),
        "lz": np.random.uniform(10.0, 20.0),
        "gap_size": np.random.uniform(5.0, 10.0),
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": np.random.choice([-1, -0.8, +0.4, +1]),
        "pw_error": 1e-6,
        "charges": [-1, +1]
    }
    params["positions"] = [
            np.array([
                np.random.uniform(0.1, params["lx"]),
                np.random.uniform(0.1, params["ly"]),
                np.random.uniform(0.1, params["lz"] - params["gap_size"]),
                ]) for _ in range(len(params["charges"]))
        ]
    params_sets.append(params)


param_sweep_accuracy_convergence(system, params_sets, accuracies = np.logspace(-1, -10, num=10))

"""
Für rdm-param bekomme ich err=1e-5, 1e-6. Espressos ELC bekommt 1e-8, 1e-9
    * reicht das an genauigkeit? (es ist sehr aufwendig die quelle von Fehlern von nur 1e-6/7 zu finden und zu fixen)


ELCIC Implementation Checkpoints
* commit="err=1e-8 for large gap sizes, swapped "f_max" formula" 1/1m err=1e-8 - BEST AT RDM PARAM (with part.Z >> 0)
* commit="fix elcic impl for no plates at all": 1/1, err=1e-1 err_for_near_plate_particles=0.4 - BEST AT Z-SHIFTING (part.Z -> 0) 



AHEAD TESTING for rdm param
* "delta_mid_top":  0, "delta_mid_bot": -1, "charges": [+1, -1], err=1e-5 (comparable to legacy)
* "delta_mid_top": -1, "delta_mid_bot": -1, "charges": [+1, -1], err=1e-2 (comparable to legacy)
    * requires "const_pot"=True in legacy_elc_energy.py
* "delta_mid_top": +1, "delta_mid_bot": -1, "charges": [+1, -1], err=1e-1 (comparable to legacy)
* "delta_mid_top": +1, "delta_mid_bot": +1, "charges": [+1, -1], err=1e-4 (better to legacy)

* "delta_mid_top": 0.3, "delta_mid_bot": -0.4, "charges": [+1, -1], err=1e-6 (better to legacy) 

* "delta_mid_top": 0.3, "delta_mid_bot": -0.4, "charges": [+1, -1, -1], err=1e-5 (better to legacy) 
    * no legacy energy: "ELC does not currently support non-neutral systems with a dielectric contrast.. Skipping..."

GRAINS OF SALT
* i compared it to the ana_sol, idk how accurate the ana_sol is
* did only a few samples per category


TODO how do i make sure i can actually "move on". i dont want to impl it in c++, realize sth isnt working and have to come back to the python prototype
"""
