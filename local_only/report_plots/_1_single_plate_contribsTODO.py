import copy

import espressomd
import numpy as np

from common.legacy.energy import get_legacy_energy
from common.plotting.param_lerp_energy_plot import run_lerp_plot
from elcic.energy._2_single_plate_neutral_non_metallic.reference_method.get_ewald2d_elcic import (
    get_ewald2d_elcic,
)
from elcic.energy.get_custom_elcic_energy import (
    get_elcic_energy,
    get_elcic_energy_contribs,
)

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 1]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([6, 5, 4]), np.array([3, 2, 4])]


def get_reference(system, params):
    return get_ewald2d_elcic(params)


def get_reference_error(system, params):
    return np.abs(get_elcic_energy(system, params) - get_reference(system, params))


def get_legacy_error(system, params):
    return np.abs(get_elcic_energy(system, params) - get_legacy_energy(system, params))


def get_E_total(system, params):
    return get_elcic_energy_contribs(system, params)["E_total"]


def get_E_near(system, params):
    return get_elcic_energy_contribs(system, params)["E_near"]


def get_E_lt(system, params):
    return get_elcic_energy_contribs(system, params)["E_lt"]


def get_E_pm1(system, params):
    return get_elcic_energy_contribs(system, params)["E_pm1"]


def get_E_l0(system, params):
    return get_elcic_energy_contribs(system, params)["E_l0"]


def get_E_far(system, params):
    return get_elcic_energy_contribs(system, params)["E_far"]


# Energy metrics (Primary)
plot_metrics = {
    ("Legacy", (("color", "green"), ("linestyle", ":"))): get_legacy_energy,
    ("E_total", (("color", "blue"), ("linestyle", "-"))): get_E_total,
    ("E_near", (("color", "blue"), ("linestyle", "-"))): get_E_near,
    ("E_lt", (("color", "blue"), ("linestyle", "-"))): get_E_lt,
    ("E_pm1", (("color", "blue"), ("linestyle", "-"))): get_E_pm1,
    ("E_l0", (("color", "blue"), ("linestyle", "-"))): get_E_l0,
    ("E_far", (("color", "blue"), ("linestyle", "-"))): get_E_far,
}

# Error metrics (Secondary)
error_metrics = {
    ("|Custom-Ref|", (("color", "purple"), ("linestyle", "--"))): get_reference_error,
    ("|Custom-Legacy|", (("color", "orange"), ("linestyle", "-."))): get_legacy_error,
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=3,
    plot_metrics=plot_metrics,
    error_metrics=error_metrics,
)

"""
* show E_contribs
* remove legend, fix labels, colors, etc.
"""
