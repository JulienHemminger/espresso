import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 3
params_sets = []

for i in range(params_count):
  
    params = {
        "lx": 22.553239340502152,
        "ly": 41.839080168737745,
        "lz": 35.51793215353896,
        "gap_size": 32.234687817568656,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-8,
        "charges": [+1, -1],
    }

    params["positions"] = [
        np.array([
            np.linspace(1.0, 8.1454445, params_count)[i],
            np.linspace(1.0, 34.44014336, params_count)[i]  ,
            np.linspace(1.0, 0.23335833, params_count)[i]  
          
        ]),
        np.array([
            np.linspace(2.0, 11.20275155, params_count)[i],
            np.linspace(2.0, 23.29430907, params_count)[i]  ,
            np.linspace(2.0, 0.7380634, params_count)[i]  
          
        ])
    ]

    params_sets.append(params)

param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 11)])

"""
===params that cause errors===
* positions, (especially p1.z, p2.z)


increasing gap_size increases error
* f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size): max_err = 1
* f_max = max(fx_max, fy_max): max_err=1e-3
    * mesh_size = p3m.get_params()["mesh"]
    * fx_max = mesh_size[0] / (2.0 * lx)
    * fy_max = mesh_size[1] / (2.0 * ly)
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
