import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points
from elc.src.energy.get_elc_energy import get_elc_energy, get_elc_energy_contribs
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
import matplotlib.pyplot as plt
from scipy.stats import linregress

from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from elc.src.forces.get_legacy_forces import get_legacy_forces
from elc.src.common.set_utils import are_sets_equal

def test_accuracy_convergence():
    l_x, l_y = 200.0, 200.0
    l_z = 10.0
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    
    pw_err = 1e-4
    gap_size = 1.0
    
    pos1, pos2 = get_rdm_constrained_points(l_x, l_y, l_z-gap_size-1e-3)
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([9.0, 9.0, 2.0])
    q1, q2 = +1.0, -1.0
    
    system.part.clear()
    system.part.add(pos=pos1, q=q1)
    system.part.add(pos=pos2, q=q2)
    
    
    analytical_forces = [(f := (q1 * q2 / np.linalg.norm(pos1 - pos2)**3) * (pos1 - pos2)), -f] # todo: any particle count
    
    legacy_forces = get_legacy_forces(system, gap_size, pw_err)
    
    assert are_sets_equal(analytical_forces, legacy_forces, tol=1e2*pw_err)
    for f in legacy_forces:
        print(f"{str(f)}")
        
    print("=========")
    for f in analytical_forces:
        print(f"{str(f)}")    
        