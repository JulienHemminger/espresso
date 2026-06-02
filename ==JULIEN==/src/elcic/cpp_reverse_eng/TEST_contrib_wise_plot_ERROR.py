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
from elcic.cpp_reverse_eng.CUSTOM.custom import get_elcic_energy
from elcic.cpp_reverse_eng.param_lerp_contrib_plot_2d import run_lerp_plot
from elcic.cpp_reverse_eng.legacy.get_legacy_contribs import get_legacy_contribs

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

