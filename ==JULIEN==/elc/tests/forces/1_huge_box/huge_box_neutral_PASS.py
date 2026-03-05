import math

from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.common.set_utils import are_sets_equal
from elc.src.forces.get_legacy_forces import get_legacy_forces
from elc.src.forces.get_elc_forces import get_elc_forces
import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.src.forces.third_party.get_analytical_dipole_forces import get_analytical_forces


def test_accuracy_convergence():
    l_x, l_y = 200.0, 200.0
    l_z = 10.0
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    pw_err = 1e-6
    gap_size = 1.0

    pos1, pos2, pos3 = get_rdm_constrained_points_np(
        l_x, l_y, l_z - gap_size - 1e-3, 3, max_distance=10
    )

    system.part.clear()
    system.part.add(pos=pos1, q=-1)
    system.part.add(pos=pos2, q=-1)
    system.part.add(pos=pos3, q=+2)

    analytical_forces = get_analytical_forces(system)

    legacy_forces = get_legacy_forces(system, gap_size, pw_err)
    elc_forces = get_elc_forces(system, gap_size, pw_err)

  
    assert are_sets_equal(analytical_forces, legacy_forces, tol=1e2*pw_err)
    assert are_sets_equal(analytical_forces, elc_forces, tol=1e2*pw_err)

