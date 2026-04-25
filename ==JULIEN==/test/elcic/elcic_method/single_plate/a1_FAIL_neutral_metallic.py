import numpy as np
import espressomd

from test.elcic.ana_method.param_sweep_accuracy_convergence import param_sweep_accuracy_convergence

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01


params_count = 1
params_sets = []
for _ in range(params_count):
    params = {
        "lx": 17, # [1, inf]
        "ly": 20, # [1, inf]
        "lz": 19, # [1, inf]
        "gap_size": 10, # [1, lz]
        "prefactor": 1.0, # [1, 5]
        "delta_mid_top": 0.0, # [-1, +1]
        "delta_mid_bot": -1.0, # [-1, +1]
        "pw_error": 1e-8, # fixed
        "charges": [+1, -1], # fixed
    }
    params["positions"] = [np.array([ 9, 20, 3]), np.array([ 1, 14, 4])] # withing [0, lx] x [0, ly] x [0, lz-gap_size] - keep a small distance eps=1e-5 to the border
        
    params_sets.append(params)

param_sweep_accuracy_convergence(system, params_sets, accuracies = [10**-i for i in range(1, 8)])

"""
test error correlation with param "lx":
* try these three parameter sets
    * default_params + lx=17 (default)
    * default_params + lx=16 (sample in negative direction)
    * default_params + lx=18 (sample in positive direction)
test error correlation with param "delta_mid_bot":
    * default_params + delta_mid_bot=-1.0 (default)
    * default_params + delta_mid_bot=-0.8 (sample in positive direction)
    * default_params + lx=-0.6 (sample in positive direction - i cant go more negative since -1 is the limit)

    
do this for all parameters (except the fixed ones)
plot the custom_error and legacy_error for each calculation

create three plots
* plot 1: the curves with the varying parameter being lx, ly, lz, gap_size
* plot 2: the curves with the varying parameter being prefactor, delta_mid_top, delta_mid_bot
* plot 3: the varying parameters are particle1.pos.x, particle1.pos.y, ... particle2.pos.z



i have paramsA where err=1e-6.
* vary params to see how the err(params) scales
    * von hand:
    * neues skript: yes

    

i have paramsA where err=1e-6.
* vary params to see how the err(params) scales


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
    * fixed params (no no.random.uniform)
    * share p3m params



"""
