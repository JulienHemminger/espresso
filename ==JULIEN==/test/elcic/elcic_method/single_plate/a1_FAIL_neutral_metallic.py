import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 3
params_sets = []

for i in range(params_count):
    lx = np.random.uniform(10.0, 50.0)
    ly = np.random.uniform(10.0, 50.0)
    lz = np.random.uniform(10.0, 50.0)
    gap_size = np.random.uniform(10.0, lz-1)

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
===params that cause errors===
* p1.z, p2.z


increasing gap_size increases error
* f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size): max_err = 1
* f_max = max(fx_max, fy_max): max_err=1e-3
    * mesh_size = p3m.get_params()["mesh"]
    * fx_max = mesh_size[0] / (2.0 * lx)
    * fy_max = mesh_size[1] / (2.0 * ly)
"""


"""
varying a single param seems mostly okay, maybe its combs of params that are causing errors?



* prefactor != 1
* delta mid top != 0

"""

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
    * is the tyagi paper wrong?
        * try: llm but no paper but online search?
    * share p3m params



"""
