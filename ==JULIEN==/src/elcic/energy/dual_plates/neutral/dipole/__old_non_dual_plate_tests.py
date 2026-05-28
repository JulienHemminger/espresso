import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy
from common.plotting.parap_lerp_plot_3d import run_2d_contour_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
}
params["lz"] = params["gap_size"] + 20
N_per_axis = 4  # make it even, else Error: no charged particles

"""
run_2d_contour_plot(
    system=system,
    base_params=params,
    x_param_key="delta_mid_bot",
    x_values=np.linspace(-1.0, 1.0, N_per_axis),
    y_param_key="delta_mid_bot",
    y_values=np.linspace(-1.0, 1.0, N_per_axis),
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
)
"""

run_2d_contour_plot(
    system=system,
    base_params=params,
    x_param_key="lx",
    x_values=np.linspace(10, 50, N_per_axis),
    y_param_key="ly",
    y_values=np.linspace(10, 50, N_per_axis),
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
)
