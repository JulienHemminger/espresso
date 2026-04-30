import numpy as np
import espressomd
from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params_count = 10
params_sets = []

for i in range(params_count):
    lx = np.random.uniform(10.0, 50.0)
    ly = np.random.uniform(10.0, 50.0)
    lz = np.random.uniform(10.0, 40.0)
    gap_size = np.random.uniform(1.0, lz - 1)

    params = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "gap_size": gap_size,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-8,
        "charges": [+1, -1],
    }

    params["positions"] = [
        np.array([
            np.random.uniform(0.1, params["lx"] - 0.1),
            np.random.uniform(0.1, params["ly"] - 0.1),
            np.random.uniform(0.1, params["lz"] - params["gap_size"] - 0.1)
        ]) for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)


param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 11)])

"""
is it backwards compatible? on random params (single plate, neutral, metallic, dipole)
* commit="err=1e-8 for large gap sizes, swapped "f_max" formula" 1/1m err=1e-8 - BEST AT RDM PARAM (with part.Z >> 0)

* commit="fix elcic impl for no plates at all": 1/1, err=1e-1 err_for_near_plate_particles=0.4 - BEST AT Z-SHIFTING (part.Z -> 0) 

Action Tree
* write test cases



* Goal: elcic.py that passes rdm_params.py and z_shifting.py


"""

"""
META PLOT (wide range of rdm params)
* see error, pick those params
* change elcic to fit on those params
* run meta-plot again
    * if i fixed the outlier and the error of other curves is still fine: accept change
    * if generally the performance got worse: discard change

"""
