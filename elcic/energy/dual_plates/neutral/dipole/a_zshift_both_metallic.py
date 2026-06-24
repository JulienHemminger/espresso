import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from common.legacy.energy import get_legacy_energy

from common.plotting.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


z_eps = 0.1
lz = 24
start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 14]), np.array([3, 2, 11])],
}
start_params["lz"] = start_params["gap_size"] + lz



z_end = z_eps # lz-z_eps
end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([6, 5, z_end]), np.array([3, 2, z_end])]

# z-shift test -> test masking, depending on part.z ALL particles are either in L0, L+1 or L-1
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=6, # TODO theres a 1e-1 error spike when part.pos.z = lz/2 (set e.g. steps=5)
)