import copy

import espressomd
import numpy as np

from elc.force.legacy_elc_forces import get_legacy_forces
from elcic.force.CUSTOM.elcic_forces import get_elcic_forces
from elcic.force.param_lerp_force_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

z_eps = 0.1
lz = 24
start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, z_eps]), np.array([3, 2, z_eps])],
}
start_params["lz"] = start_params["gap_size"] + lz



z_end = lz - z_eps
end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([6, 5, z_end]), np.array([3, 2, z_end])]


run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_force=get_elcic_forces,
    get_legacy_force=get_legacy_forces,
    get_analytical_force=None,
    steps=6,
)