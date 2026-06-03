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
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from elcic.cpp_reverse_eng.param_lerp_contrib_plot_2d import run_lerp_plot
from elcic.cpp_reverse_eng.legacy.get_legacy_contribs import get_legacy_contribs

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 4])], # legacy fails for part.z <= 3    
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 15.0, # legacy runs with: 14, 15, fails with 16, 20
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 30]), np.array([3, 2, 30])], # legacy runs for part.z = 4, 24, fails for part.z = 3, 34, 39    
}
end_params["lz"] = start_params["gap_size"] + 40





system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


run_lerp_plot(system=system, start_params=start_params, end_params=end_params, get_custom_energy=get_elcic_energy, get_legacy_energy=get_legacy_contribs, steps=6)

"""
Ich schaffe es nicht meine custom python elcic methode so zu verändern dass der fehler bzgl. der einzelnen Contributions unter ca. 1e-3 ist.
Mit den "ground truth" Contributions aus elc.cpp lässt sich der Fehler zwar ein bisschen eingrenzen (in E_far oder E_near, das sind beides jeweils noch ca. 100 Zeilen an Code) aber es ist immernoch kaum möglich herauszufinden wo/was genau der Fehler ist um ihn zu beheben.




Außerdem bin ich mir nicht sicher meine Annahme welche Zwischenergebnissen/Variablen aus elc.cpp zu welchen Contribution-Termen aus dem Paper (z.B. Φ(L0​,LT​)) gehören, so ganz stimmt.

Hier z.B. kann man sehen dass bei E_total meine "Custom" Methode mit err=6e-5 ziemlich gut ist, aber die Contributions nur auf error=6e-3 kommen.

Parameters={
 'lx': 34.0,
 'ly': 34.0,
 'lz': 38.0,
 'gap_size': 13.0,
 'pw_error': 1e-08,
 'prefactor': 1.0,
 'delta_mid_top': 0.2,
 'delta_mid_bot': 0.2,
 'positions': [[6.0, 5.0, 19.6], [3.0, 2.0, 19.6]],
 'charges': [1.0, -1.0]}


Name         | Legacy     | Custom     | Error (abs)
--------------------------------------------------
e_total      | -0.2356040529   | -0.2355368480   | 0.0000672048   
e_far        | 0.0010904388    | 0.0000084821    | 0.0010819567   
e_near       | -0.2290333091   | -0.2355453302   | 0.0065120211  

* Bei Legacy ist e_total = e_far + e_near + coulomb1 (dieser term von dem wir nicht sicher sind wo der hingehört)


"""