import espressomd
import numpy as np
from elc.debug.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy
from elc.debug.analytical import (
    get_ewald_energy_2d
)
from common.plotting.param_lerp_plot_2d import run_lerp_plot
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
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
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10

run_lerp_plot(system, start_params=start_params, end_params=end_params, get_custom_energy=get_elcic_energy, get_analytical_energy=get_ewald_energy_2d, get_legacy_energy=get_legacy_energy, steps=5)
