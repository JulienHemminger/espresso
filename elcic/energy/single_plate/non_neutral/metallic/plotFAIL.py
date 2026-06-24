import copy

import espressomd
import numpy as np
from elcic.energy.single_plate.neutral.metallic.custom import get_elcic_energy
from common.plotting.param_lerp_plot_2d import run_lerp_plot
from common.legacy.energy import get_legacy_energy

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+0.3, -0.6],
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 40


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+0.7, -0.2],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 40

"""
The standard Ewald summation (even in 3D) is mathematically ill-defined for non-neutral systems because the Coulomb potential energy of a net-charged periodic system diverges (the "monopole problem")
"""
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=None,
    steps=10,
)


"""
legacy: doent work for systems like this


fix ana sol:
* extend ewald method: NO, doesnt work for systems like this


fix custom (focus on custom-legacy)
"""