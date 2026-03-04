import pytest
import numpy as np
import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points
from elc.src.energy.get_elc_energy import get_elc_energy
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy

def test_dipole_rdm_pos_energy_non_square():
    # --- Setup ---
    l_x, l_y, l_z = 100.0, 50.0, 10.0
    pw_error = 1e-6
    gap_size = 1.0
    
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    # Initialize P3M
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # --- Simulation State ---
    # Testing with 2 particles as per original logic
    point_count = 2
    system.part.clear()

    # Generate positions and charges
    rs = get_rdm_constrained_points(l_x, l_y, l_z - gap_size - 1e-3, point_count)
    qs = [+1.0, -1.0] 
    
    for i in range(min(len(rs), len(qs))):
        system.part.add(pos=rs[i], q=qs[i])

    # --- Energy Calculation ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    # Logic: Difference should be within 1e2 * pw_error
    diff = abs(legacy_energy - elc_energy)
    tolerance = 1e2 * pw_error
    
    assert diff <= tolerance, (
        f"Energy mismatch too high! "
        f"Legacy: {legacy_energy}, ELC: {elc_energy}, Diff: {diff}, Tol: {tolerance}"
    )