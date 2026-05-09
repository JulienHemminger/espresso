import numpy as np
import espressomd
from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence
import random

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params_count = 5
params_sets = []

for i in range(params_count):

    params = {
        "lx": np.random.uniform(10.0, 50.0),
        "ly": np.random.uniform(10.0, 50.0),
        "lz": np.random.uniform(10.0, 40.0),
        "gap_size": np.random.uniform(10.0, 20.0),
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-8,
        "charges": [+1, -1],
    }

    eps = 1e-2
    params["positions"] = [
        np.array([
            np.random.uniform(eps, params["lx"] - eps),
            np.random.uniform(eps, params["ly"] - eps),
            np.random.uniform(eps, params["lz"] - params["gap_size"] - eps)
        ]) for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)


param_sweep_accuracy_convergence(system, params_sets, accuracies = np.logspace(-1, -10, num=10))
