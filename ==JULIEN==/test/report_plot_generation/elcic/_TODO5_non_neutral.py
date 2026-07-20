import espressomd
import numpy as np
from src.common.plotting.param_lerp_energy_plot import run_lerp_plot
from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from src.elcic.energy.single_plate.neutral.metallic.analytical import (
    get_ewald_elcic_2d as get_ewald2d_elcic,
)

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)
import copy

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -0.7, -0.3],
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([4, 2, 6])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 10

end_params = copy.deepcopy(start_params)
end_params["charges"] = [+1.0, -1.0, -1.0]


def get_custom_energy(system, params):
    return get_elcic_energy(system, params)["e_total"]


def eval_reference(system, params):
    return get_ewald2d_elcic(params)


def get_ref_error(system, params):
    return np.abs(get_custom_energy(system, params) - eval_reference(system, params))


# Energy metrics (Primary)
plot_metrics = {
    ("Custom ELCIC", (("color", "blue"), ("linestyle", "-"))): get_custom_energy,
    ("Ewald 2D", (("color", "red"), ("linestyle", "-"))): eval_reference,
}

# Error metrics (Secondary)
error_metrics = {
    ("|Custom-Ewald2D|", (("color", "orange"), ("linestyle", "-."))): get_ref_error,
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=4,
    plot_metrics=plot_metrics,
    error_metrics=error_metrics,
)

"""
cant use legacy ELCIC here.
"""
import espressomd.electrostatics

espressomd.electrostatics.MMM1D
