import espressomd
import numpy as np

from common.legacy.forces import get_legacy_forces
from common.plotting.param_lerp_force_plot import run_lerp_plot
from elcic.force.get_custom_elcic_forces import get_elcic_forces

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "lambda": 20,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
}
start_params["lz"] = start_params["gap_size"] + 40


end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "lambda": 1e-3,
    "positions": [np.array([4, 5, 6]), np.array([1, 2, 3])],
}
end_params["lz"] = start_params["gap_size"] + 10


run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_force=get_elcic_forces,
    get_legacy_force=get_legacy_forces,
    get_analytical_force=None,
    steps=6,
)
