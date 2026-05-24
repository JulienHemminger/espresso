import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.non_metallic.sweep_pos_z.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from elcic.energy.dual_plates.neutral.non_metallic.param_lerp_plot_2d import run_lerp_plot

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
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10



run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=6,
)


"""
steps=odd: fails, "No charged particles in system"
steps=even: works,


Action Tree
* wtf are those "hard" right errors in "lerp2d_2026-05-22_17-01-35"
    * why is there an abrupt hole? is this right?

* imrpove errors show in "lerp2d_2026-05-22_17-01-35"
    * fix the large 1e-1 errors at t->1

    * improve remaining (right now err=1e-2 - 1e-4, - id like 1e-8)
        * far formula has bug where it treats plates differently 




* fix part.z = lz/2 spikes/errors
* rn i have multiple custom.py. i want a single one that passes all tests

"""