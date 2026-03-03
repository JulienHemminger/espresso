import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.get_elc_energy import get_elc_energy
from elc.src.third_party.get_legacy_elc import get_legacy_elc_energy
from common.get_positions import get_rdm_constrained_points
from elc.src.third_party.get_ewald_energy_2d import get_ewald_energy_2d

@pytest.fixture(scope="module")
def system():
    """Persistent Espresso system for triplet tests."""
    s = espressomd.System(box_l=[100.0, 100.0, 10.0])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

def test_non_neutral_tripole(system):
    # --- Configuration ---
    l_x, l_y, l_z = 100.0, 100.0, 10.0
    gap_size = 1.0
    pw_error = 1e-6
    charges = [1.0, -1.0, -1.0] # Net charge = -1.0
    
    system.part.clear()
    system.box_l = [l_x, l_y, l_z]
    positions = get_rdm_constrained_points(l_x, l_y, l_z-gap_size-1e-3, point_count=3)

    # --- Add Particles & Solver ---
    for i in range(3):
        system.part.add(pos=positions[i], q=charges[i])

    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # --- Calculation ---
    ana_energy = get_ewald_energy_2d(system)
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    # Tolerance for Non-Neutral systems (against analytical) is higher (~0.1)
    # Consistency between ELC methods should still be tight (~1e-5)
    max_error = 1e2 * pw_error
    assert elc_energy == pytest.approx(ana_energy, abs=max_error), \
        f"Triplets vs Analytical failed: ELC={elc_energy}, Analytical={ana_energy}"
        
    assert elc_energy == pytest.approx(legacy_energy, abs=1e-5), \
        f"New ELC vs Legacy ELC mismatch: {elc_energy} vs {legacy_energy}"
        