import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 5
params_sets = []
for _ in range(params_count):
    params = {
        "lx": 17,
        "ly": 20,
        "lz": 19,
        "gap_size": np.random.uniform(8.0, 15.0),
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-8,
        "charges": [+1, -1],
        "positions": [np.array([9, 20, 3]), np.array([1, 14, 4])],
    }
    params_sets.append(params)

param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 11)])

"""
varying a single param seems mostly okay, maybe its combs of params that are causing errors?


===params that cause errors===
* prefactor != 1
* delta mid top != 0



* gap size > 10
    * error increases with increasing gap_size
* p1.z or p2.z < 4

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
