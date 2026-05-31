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
from elcic.energy.legacy_contrib_framework.get_legacy_contribs import get_legacy_contribs

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 4])], # legacy fails for part.z <= 3    
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 15.0, # legacy runs with: 14, 15, fails with 16, 20
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 30]), np.array([3, 2, 30])], # legacy runs for part.z = 4, 24, fails for part.z = 3, 34, 39    
}
end_params["lz"] = start_params["gap_size"] + 40





system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


run_lerp_plot(system=system, start_params=start_params, end_params=end_params, get_custom_energy=get_elcic_energy, get_legacy_energy=get_legacy_contribs, steps=4)


DONE = None
f"""
Action Tree
* fix custom.py for this single test: {DONE}
* add range of test (lerp) and plots (now im at my old err=1e-4): {DONE}
* swap E_near and E_far in legacy contribs, i thing theres a bug on the C++ side: {DONE}
* break it doen to E_near_p3m or E_near_corr: {DONE}


* fix E_far
    * use Gemini, with tyagi + custom.py: NO, 10 tries 
    * cursor composer?
"""