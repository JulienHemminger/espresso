import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.versions.a_1e4 import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from elcic.energy.dual_plates.neutral.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


z_eps = 0.1
lz = 20
start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, z_eps]), np.array([3, 2, z_eps])],
}
start_params["lz"] = start_params["gap_size"] + lz



end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([6, 5, lz-z_eps]), np.array([3, 2, lz-z_eps])]


run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=5,
)


"""
* sweep



Goal:
* symmatrical energy
* custom - legacy < 1e-6


Action Tree
* direct implementation
    * LLM with tyagi: NO (15 tries)
    * LLM with elc.cpp: NO (2 tries)



HANDLE OTHER SYSTEM (non neutral, etc) FIRST

FIND OTHER PARAMS
* delta_mid_top=0,  delta_mid_bot=any in -1 to +1: single bottom plate works, is well tested

* delta_mid_top=0.1,  delta_mid_bot=0.1: TODO weak dielectric contrast case, rapid convergence, 


* delta_mid_top=1,  delta_mid_bot=0: test single top plate
* delta_mid_top=0.1,  delta_mid_bot=0.1: 
* delta_mid_top=1,  delta_mid_bot=1: adds the divergent infinite sum


"""
