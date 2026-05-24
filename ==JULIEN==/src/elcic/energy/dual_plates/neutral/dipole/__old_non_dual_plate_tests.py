import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from elcic.energy.dual_plates.neutral.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

# https://gemini.google.com/app/5ee31aa2c36a5960

start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
}
start_params["lz"] = start_params["gap_size"] + 20



end_params = copy.deepcopy(start_params)
end_params["lambda"] = 0

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy, 
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=6,
)

