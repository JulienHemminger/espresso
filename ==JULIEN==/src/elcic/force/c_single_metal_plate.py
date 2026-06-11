import copy

import espressomd
import numpy as np

from elc.force.legacy_elc_forces import get_legacy_forces
from elcic.force.CUSTOM.elcic_forces import get_elcic_forces
from elcic.force.param_lerp_force_plot_2d import run_lerp_plot

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
    "delta_mid_top": 0.8,
    "delta_mid_bot": -0.5,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 10]), np.array([3, 2, 10])],
}
start_params["lz"] = start_params["gap_size"] + 40



end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.8,
    "delta_mid_bot": -0.5,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([4, 5, 10]), np.array([1, 2, 10])],
}
end_params["lz"] = start_params["gap_size"] + 10


run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_force=get_elcic_forces,
    get_legacy_force=get_legacy_forces,
    get_analytical_force=None,
    steps=6,
)

"""
use AI tools
* cursor composer
* via ai arena claude & chatgpt: NO
* gemini: NO? 7 tries Flash, 10 tries Flash-Lite
    * dont make LLm rewite entire function, make it contrib-ized: NO, 5 tries
    * prompt "only add bottom plate reflection" ? 
    * series of carefully selected tests TRY THIS: https://gemini.google.com/app/554fa14f93a80c7b


    
"""