import pytest
import numpy as np
import espressomd
import espressomd.electrostatics
from common.get_positions import get_rdm_constrained_points
from common.get_charges import get_rdm_charges
from elc.src.get_elc_energy import get_elc_energy
from elc.src.third_party.get_legacy_elc import get_legacy_elc_energy

@pytest.fixture(scope="module")
def system():
    """Module-level system to avoid 'one instance' RuntimeError."""
    s = espressomd.System(box_l=[10.0, 10.0, 3.0])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

@pytest.mark.parametrize("point_count", [2])
def test_energy_computation_residuals(system, point_count):
    # --- Setup ---
    l_x, l_y, l_z = 10.0, 10.0, 3.0
    pw_error = 1e-4
    gap_size = 1.0
    
    # Reset system state
    system.part.clear()
    system.box_l = [l_x, l_y, l_z]

    # Generate particles
    rs = get_rdm_constrained_points(l_x, l_y, l_z - gap_size - 1e-3, point_count)
    qs = get_rdm_charges(point_count)
    
    for i in range(min(len(rs), len(qs))):
        system.part.add(pos=rs[i], q=qs[i])

    # Initialize Solver
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # --- Calculation ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    # We check if the difference is within 1e2 * pw_error as requested
    tolerance = 1e2 * pw_error
    
    assert elc_energy == pytest.approx(legacy_energy, abs=tolerance), \
        f"Energy mismatch for {point_count} particles: Legacy={legacy_energy}, ELC={elc_energy}"