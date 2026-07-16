import copy
import random

import espressomd
import numpy as np

from common.legacy.energy import get_legacy_energy
from common.plotting.param_lerp_energy_plot import run_lerp_plot
from elcic.energy.temp._2_dual_plate_hack_get_custom_elcic_energy import (
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
    "positions": [np.array([6, 5, 1]), np.array([3, 2, 4])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([6, 5, 3]), np.array([3, 2, 9])]


def get_reference(system, params):
    return get_legacy_energy(system, params)


def get_reference_error(system, params):
    return np.abs(
        get_elcic_energy(system, params) - get_reference(system, params),
    ) + random.uniform(1e-8, 1e-6)


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


plot_metrics = {
    (
        r"Reference",
        (("color", "red"), ("linestyle", "-."), ("linewidth", 2), ("marker", "s")),
    ): get_reference,
    (
        r"$E_{total}$",
        (("color", "dodgerblue"), ("linestyle", ":"), ("linewidth", 2.5)),
    ): get_E_total,
    (r"$E_{near}$", (("color", "cyan"), ("linestyle", "--"))): get_E_near,
    (r"$E_{lt}$", (("color", "steelblue"), ("linestyle", "--"))): get_E_lt,
    (r"$E_{±1}$", (("color", "skyblue"), ("linestyle", "--"))): get_E_pm1,
    (r"$E_{l0}$", (("color", "purple"), ("linestyle", "--"))): get_E_l0,
    (r"$E_{far}$", (("color", "cyan"), ("linestyle", "-."))): get_E_far,
}

# Updated Error Metrics
error_metrics = {
    (
        r"$|E_{total} - \text{Ewald 2D}|$",
        (("color", "orange"), ("linestyle", ":")),
    ): get_reference_error,
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=20,
    plot_metrics=plot_metrics,
    error_metrics=error_metrics,
)


"""
t= 0.73


"start_positions": [np.array([6, 5, 1]), np.array([3, 2, 4])],
"end_positions" = [np.array([6, 5, 3]), np.array([3, 2, 9])]


z1 = 2.46
z2 = 7.65


z1+z2 = 10

7.65 - 2.46 = 5.19
"""


"""

t= 0.73


"start_positions": [np.array([6, 5, 1]), np.array([3, 2, 4])],
"end_positions" = [np.array([6, 5, 3]), np.array([3, 2, 9])]

$z_1 = 3.19$
$z_2 = 7.38$

z2-z1 = 4.19
"""
