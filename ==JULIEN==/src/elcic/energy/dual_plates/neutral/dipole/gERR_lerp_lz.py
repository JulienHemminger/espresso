import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from common.plotting.param_lerp_plot_2d import run_lerp_plot

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
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
}
start_params["lz"] = start_params["gap_size"] + 40

end_params = copy.deepcopy(start_params)
end_params["lz"] = start_params["gap_size"] + 20



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
* is legacy correct?



Custom = E_near = E_near bot
(E_near_top = 0)

od i need to fix E_near_bot, or do i need e_near_top to "balance it out"?




Large (1e-1) error when lerp "positions".
* start_pos = [np.array([1, 2, 3]), np.array([4, 5, 6])],
    * end_pos = start_pos, err_right=1e-2
    * end_pos = [np.array([6, 5, 6]), np.array([3, 2, 1])], err_right=1e-1

    * end_pos = start_pos + p1.z=19, err_right=1e-1
    * end_pos = start_pos + p2.z=19, err_right=1e-1



    
* lerp pos.x, y, z individually




right valley happens for
* all param change err_right=1e0
* all except lx ly, err_righh=1e-1 (smaller but significant)
* all except lx ly, lz, gap_size, err_righh=1e-1 (smaller but significant)


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