import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 20
params_sets = []

for i in range(params_count):
    lx = np.random.uniform(10.0, 50.0)
    ly = np.random.uniform(10.0, 50.0)
    lz = np.random.uniform(10.0, 40.0)
    gap_size = np.random.uniform(10.0, 20.0)
    pw_error = 10**np.random.uniform(-8, -3) 
    delta_top = 0.0 #np.random.uniform(-1.0, 1.0)
    delta_bot = np.random.uniform(-1.0, 1.0)

    params = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "gap_size": gap_size,
        "prefactor": 1.0,
        "delta_mid_top": delta_top,
        "delta_mid_bot": delta_bot,
        "pw_error": pw_error,
        "charges": [+1, -1],
    }

    params["positions"] = [
        np.array([
            np.random.uniform(0.1, params["lx"] - 0.1),
            np.random.uniform(0.1, params["ly"] - 0.1),
            np.random.uniform(0.1, params["lz"] - 0.1)
        ]) for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)

param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 11)])

"""
* try a bunch of different f_max=... formulas


increasing gap_size increases error
* f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size): max_err = 1e0
* f_max = max(fx_max, fy_max): max_err=1e-3
    * mesh_size = p3m.get_params()["mesh"]
    * fx_max = mesh_size[0] / (2.0 * lx)
    * fy_max = mesh_size[1] / (2.0 * ly)
"""


"""
varying a single param seems mostly okay, maybe its combs of params that are causing errors?


===params that cause errors===
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
