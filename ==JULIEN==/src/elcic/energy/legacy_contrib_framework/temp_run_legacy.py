import sys
import re
import os
from elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
import math

import sys
import io
import re
import copy
import os
import sys
import io
import re
import espressomd
import numpy as np
from elcic.energy.legacy_contrib_framework.CUSTOM.custom import get_elcic_energy
from elcic.energy.legacy_contrib_framework.param_lerp_plot_2d import run_lerp_plot
from elcic.energy.legacy_contrib_framework.get_legacy_contribs import get_legacy_contribs

params = {
    "lx": 30.0,
    "ly": 30.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([4, 5, 6]), np.array([1, 2, 3])], # legacy fails for part.z <= 3    
}
params["lz"] = params["gap_size"] + 30

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)
system.electrostatics.clear()
        
system.part.clear()
system.box_l = [params["lx"], params["ly"], params["lz"]]
for i in range(len(params["charges"])):
    system.part.add(pos=params["positions"][i], q=params["charges"][i])
print("=======================")
print(params)
print(f"E_return = {get_legacy_energy(system, params)}")

"""
ELC] E_total = -0.276515461166035
E_return = -0.2765154822643574
"""
