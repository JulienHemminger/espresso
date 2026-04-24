import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 1
params_sets = []
for _ in range(params_count):
    params = {
        "lx": np.random.uniform(10.0, 20.0),
        "ly": np.random.uniform(10.0, 20.0),
        "lz": np.random.uniform(10.0, 20.0),
        "gap_size": np.random.uniform(5.0, 10.0),
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": 0.0,
        "pw_error": 1e-4,
        "charges": [+1, -1],
    }
    params["positions"] = [
            np.array([
                np.random.uniform(0.1, params["lx"]),
                np.random.uniform(0.1, params["ly"]),
                np.random.uniform(0.1, params["lz"] - params["gap_size"]),
                ]) for _ in range(len(params["charges"]))
        ]
    params_sets.append(params)

param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 8)])

"""    

* start with blank custom_elcic: DONE

* refac/debug/fix/test individual contribs one after another: TODO

    * add plots for indiv contribs

    * no plates at all (no reflec contrib, only p3m?): 
    * large box (no pbc images contrib)


* other ideas
    * share p3m params



"""
