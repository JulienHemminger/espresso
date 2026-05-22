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
f"""
whats the problem
* with mask conditions: error right, legacy=+4, custom=0
* with hack conditions: good, err=1e-4





Action Tree
* fix near (major contrib 5)
    * replace the hack fix with "physical" logic
        * LLM + Tyagi + "1e-3 for all db+-1 dt+-1_combis.py": {NO}, 4 tries

        * by hand (i think i just need to filter particles in L-1, L0, L+1, etc) 
            * basically replace "set to 0" by "set part=empty"
            * kinda like sonnet in https://arena.ai/c/019e4ef2-a1e3-79ec-941d-baa716843c49




        

* fix far (minor contrib 1e-5) - asymmetric "if np.any(m_top):" in get_far_field_energy




"""
