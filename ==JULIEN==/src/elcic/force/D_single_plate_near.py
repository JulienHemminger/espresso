import copy

import espressomd
import numpy as np

from elc.force.legacy_elc_forces import get_legacy_forces
from elcic.force.CUSTOM.elcic_forces import get_elcic_forces
from elcic.force.param_lerp_force_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4


lz = 40
eps = 1
start_params = {
    "lx": 50.0, 
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -0.5,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "lambda": lz/4,
    "positions": [np.array([6, 5, 0+eps]), np.array([3, 2, 0+eps])],
}
start_params["lz"] = start_params["gap_size"] + lz


end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([4, 5, lz-eps]), np.array([1, 2, lz-eps])]


run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_force=get_elcic_forces,
    get_legacy_force=get_legacy_forces,
    get_analytical_force=None,
    steps=5,
)

"""
2_C_print_seems_correct_but_err1e-3: err=1e-1 to 1e-7
1_AB_1e-20.py: err=1e-1 to 1e-7
"""