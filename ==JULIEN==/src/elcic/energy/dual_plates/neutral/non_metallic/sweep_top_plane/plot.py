import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.non_metallic.sweep_top_plane.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from elcic.energy.dual_plates.neutral.non_metallic.param_lerp_plot_2d import run_lerp_plot

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
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
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
NO = None

"""
* stems from e_near (e_far is only 1e-6)
with phys mask: err=1e-1 in center, 1e-4 at edge
with hack mask: err=1e-4 everywhere
* tuning lambda?
    * lambda=2.0, error_center=1e-1
    * lambda=4.0, error_center=1e-1
    * lambda=8.0, error_center=1e-7 (with a center 1e-1 spike)
    * lambda=lz/2, error_center=1e-7 (with a center 1e-1 spike)

    
* spike only when part.z = lz/2 (exactly middle)


Action Tree
* fix the part.z = lz/2 spike now {NO}
* continue (param sweep, etc): {YES}

* fix near (major contrib 5): DONE


        

* fix far (minor contrib 1e-5) - asymmetric "if np.any(m_top):" in get_far_field_energy




""" 
