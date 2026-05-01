import numpy as np
import espressomd
from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params_count = 5
params_sets = []

d_pos = 1e-2
min_l = 25
max_l = 50

for i in range(params_count):
    lx = np.random.uniform(min_l, max_l)
    ly = np.random.uniform(min_l, max_l)
    lz = np.random.uniform(min_l, max_l)
    gap_size = np.random.uniform(10.0, 25.0)

    params = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "gap_size": gap_size,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-6,
        "charges": [+1, -1],
    }

    params["positions"] = [
        np.array([
            np.random.uniform(d_pos, params["lx"] - d_pos),
            np.random.uniform(d_pos, params["ly"] - d_pos),
            np.random.uniform(d_pos, params["lz"] - params["gap_size"] - d_pos)
        ]) for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)


param_sweep_accuracy_convergence(system, params_sets, accuracies = np.logspace(-1, -10, num=10))

"""
ELCIC Implementation Checkpoints
* commit="err=1e-8 for large gap sizes, swapped "f_max" formula" 1/1m err=1e-8 - BEST AT RDM PARAM (with part.Z >> 0)
* commit="fix elcic impl for no plates at all": 1/1, err=1e-1 err_for_near_plate_particles=0.4 - BEST AT Z-SHIFTING (part.Z -> 0) 

"""
