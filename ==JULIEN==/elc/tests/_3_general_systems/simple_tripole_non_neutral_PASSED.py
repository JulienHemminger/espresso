import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.get_elc_energy import get_elc_energy
from elc.src.third_party.get_legacy_elc import get_legacy_elc_energy

@pytest.fixture(scope="module")
def system():
    """Persistent Espresso system for triplet tests."""
    s = espressomd.System(box_l=[100.0, 100.0, 10.0])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

def generate_triplet_test_cases(count=20, l_z=10.0, gap=1.0):
    """Generator for random triplet positions to feed into parametrization."""
    limit = l_z - gap - 0.1
    cases = []
    for _ in range(count):
        pos = [np.random.uniform(0.5, limit, 3) for _ in range(3)]
        # Basic check to avoid overlapping particles
        r12 = math.dist(pos[0], pos[1])
        r13 = math.dist(pos[0], pos[2])
        r23 = math.dist(pos[1], pos[2])
        if all(r > 0.8 for r in [r12, r13, r23]):
            cases.append(pos)
    return cases

@pytest.mark.parametrize("positions", generate_triplet_test_cases())
def test_triplet_energy_residuals(system, positions):
    # --- Configuration ---
    l_xy, l_z = 100.0, 10.0
    gap_size = 1.0
    pw_error = 1e-6
    charges = [1.0, -1.0, -1.0] # Net charge = -1.0
    
    system.part.clear()
    system.box_l = [l_xy, l_xy, l_z]

    # --- Analytical Baseline ---
    r12 = math.dist(positions[0], positions[1])
    r13 = math.dist(positions[0], positions[2])
    r23 = math.dist(positions[1], positions[2])
    
    ana_energy = (charges[0]*charges[1])/r12 + \
                 (charges[0]*charges[2])/r13 + \
                 (charges[1]*charges[2])/r23

    # --- Add Particles & Solver ---
    for i in range(3):
        system.part.add(pos=positions[i], q=charges[i])

    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # --- Calculation ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    # Tolerance for Non-Neutral systems (against analytical) is higher (~0.1)
    # Consistency between ELC methods should still be tight (~1e-5)
    
    assert elc_energy == pytest.approx(ana_energy, abs=0.15), \
        f"Triplets vs Analytical failed: ELC={elc_energy}, Analytical={ana_energy}"
        
    assert elc_energy == pytest.approx(legacy_energy, abs=1e-5), \
        f"New ELC vs Legacy ELC mismatch: {elc_energy} vs {legacy_energy}"