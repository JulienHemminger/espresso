import espressomd
import numpy as np
from src.common.plotting.param_lerp_energy_plot import run_lerp_plot
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from src.elcic.energy.single_plate.neutral.metallic.analytical import (
    get_ewald_elcic_2d as get_ewald2d_elcic,
)

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
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 40

end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10


def eval_reference(system, params):
    return get_ewald2d_elcic(params)


def get_ref_error(system, params):
    return np.abs(get_elcic_energy(system, params) - eval_reference(system, params))


def get_legacy_error(system, params):
    return np.abs(get_elcic_energy(system, params) - get_legacy_energy(system, params))


# Energy metrics (Primary)
plot_metrics = {
    ("Legacy", (("color", "green"), ("linestyle", ":"))): get_legacy_energy,
    ("Custom", (("color", "blue"), ("linestyle", "-"))): get_elcic_energy,
}

# Error metrics (Secondary)
error_metrics = {
    ("|Custom-Ref|", (("color", "purple"), ("linestyle", "--"))): get_ref_error,
    ("|Custom-Legacy|", (("color", "orange"), ("linestyle", "-."))): get_legacy_error,
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=10,
    plot_metrics=plot_metrics,
    error_metrics=error_metrics,
)
