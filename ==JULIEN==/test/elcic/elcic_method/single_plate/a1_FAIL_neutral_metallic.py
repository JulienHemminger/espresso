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
        "pw_error": 1e-6,
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
ALL-AT-ONCE DEVELOPMENT (ELC -> ELCIC)
* start at regular elc: DONE
    * add single, diel interface: NO - 20 tries and it didnt work
    * try other params(two plates, different deltas): NO

    * prompt explicitly for single metallic plate (delta_mid_bot=-1), no inf reflections: 

* start at minimal elc (no non neutral corr)
    * ...


COMPONENT WISE DEVELOPMENT (ELC -> ELC with .. term -> ...)
* start with blank custom_elcic
* refac/debug/fix/test individual contribs one after another: TODO

    * what is the only contrib affected if i put delta_mid_bot=-1.0
        * add plot for that contrib
        * impl that contrib
            * IDEA: first compute it brute force, test it, then impl as more efirricent elcic code
        * test that contrib



    * 

    * add plots for indiv contribs

    * large box (no pbc images contrib)


* other (not very promising) ideas
    * share p3m params



"""
