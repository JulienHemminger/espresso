import copy

import espressomd
import numpy as np
from elcic.energy.get_custom_elcic_energy import get_elcic_energy
from common.legacy.energy import get_legacy_energy

from elcic.energy._2_single_plate_neutral_non_metallic.reference_method.get_ewald2d_elcic import (
    get_ewald2d_elcic,
)
from common.plotting.param_lerp_energy_plot import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": -1.0,
    "charges": [+0.4, -0.1, +0.9, -0.6],
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([3, 2, 1]), np.array([4, 3, 1])],
    "pw_error": 1e-8,
    "lambda": 1e-1,
}
start_params["lz"] = start_params["gap_size"] + 40

end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": +1.0,
    "charges": [+0.7, -0.8, +0.2, -0.1],
    "positions": [np.array([6, 5, 5]), np.array([3, 4, 4]), np.array([1, 1, 2]), np.array([2, 3, 3])],
    "pw_error": 1e-8,
    "lambda": 5,
}
end_params["lz"] = end_params["gap_size"] + 10

def eval_reference(system, params):
    return get_ewald2d_elcic(params)

plot_metrics = {
    ("Custom", (("color", "blue"), ("linestyle", "-"), ("linewidth", 2))): get_elcic_energy,
    ("Reference", (("color", "red"), ("marker", "o"), ("linestyle", "None"))): eval_reference
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=10,
    plot_metrics=plot_metrics,
)