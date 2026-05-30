import sys
import io
import re
import copy
import os
import sys
import io
import re
import espressomd
import numpy as np
from elcic.energy.legacy_contrib_framework.CUSTOM.custom import get_elcic_energy
from elcic.energy.legacy_contrib_framework.param_lerp_plot_2d import run_lerp_plot
from elc.energy.legacy_elc_energy import get_legacy_energy
from elcic.energy.legacy_contrib_framework.get_legacy_contribs import get_legacy_contribs

"""
eps = 1e-1
lz = 10
start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, eps]), np.array([3, 2, eps])],    
}
start_params["lz"] = start_params["gap_size"] + lz

lz = 40
end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, lz-eps]), np.array([3, 2, lz-eps])],    
}
end_params["lz"] = start_params["gap_size"] + lz
"""
eps = 4 # somehow legacy throws errors for eps=2, 3
lz = 10
start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, eps]), np.array([3, 2, eps])],    
}
start_params["lz"] = start_params["gap_size"] + lz


end_params = copy.deepcopy(start_params)
end_params["delta_mid_bot"] = 1.0




system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


run_lerp_plot(system=system, start_params=start_params, end_params=end_params, get_custom_energy=get_elcic_energy, get_legacy_energy=get_legacy_contribs, steps=4)



"""
Action Tree
* fix custom.py for this single test
* add range of test (lerp) and plots (now im at my old err=1e-4): DONE
* swap E_near and E_far in legacy contribs, i thing theres a bug on the C++ side: DONE

* TODO why do i get errors for some params?
* TODO fix E_near
    * can/should i break it doen to E_near_p3m or E_near_corr.
"""