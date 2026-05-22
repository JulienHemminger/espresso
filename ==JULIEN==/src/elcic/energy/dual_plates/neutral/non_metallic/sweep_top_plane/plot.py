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
    "delta_mid_top": -1.0,
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
f"""
what causes the error?
* lerping z positions
    * db=-1, dt=+1, err= 8 left sign swap(near bottom plate)
    * db=+1, dt=+1, err=1e-4
    * db=-1, dt=-1, err=1e-4

    * db=+1, dt=-1, err= 8 left sign swap(near bottom plate)
        * at bot plate: legacy: +4, custom: -4


Action Tree
* fix all at once: {NO}
    * LLM left_swapped_sign, treat top and bot plane the same; {NO}, 15 tries

    * debug by hand? just do same stuff for delta_top as delta_bot {NO}
        * remove unusual delta_mid_top logic in "get_elcic_energy": {NO}, worse error
        * remove unusual delta_mid_top logic in  "_get_far_field_energy": {NO} no cange

        
* fix near (major contrib 5)

* fix far (minor contrib 1e-5) - asymmetric "if np.any(m_top):" in get_far_field_energy





NOTE

narrow down which contrib causes the error
* _get_chi_components: not changed 
* _get_far_field_energy: not changed
* _get_config_energy: not changed
* DIFFERENCE IS IN get_elcic_energy, incl _get_interaction_energy

"""
