import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.get_elc_energy import get_elc_energy
from elc.src.get_legacy_elc import get_legacy_elc_energy

@pytest.fixture(scope="module")
def system():
    """Initialize the Espresso system once for the module."""
    s = espressomd.System(box_l=[100.0, 100.0, 10.0])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

# Defining the test cases as (pos1, pos2, q1, q2)
# You can add more tuples to this list to test different distances/charges
@pytest.mark.parametrize("pos1, pos2, q1, q2", [
    ([50.0, 50.0, 1.0], [50.0, 50.0, 9.0], +1.0, +1.0), 
])
def test_elc_vs_analytical_energy(system, pos1, pos2, q1, q2):
    # --- Setup ---
    l_xy, l_z = 100.0, 10.0
    gap_size = 1.0
    pw_error = 1e-6
    
    system.box_l = [l_xy, l_xy, l_z]
    system.part.clear()

    # Calculate analytical baseline: E = (q1 * q2) / r
    r = math.dist(pos1, pos2)
    assert r >= 1.0, "Particles are too close for stable P3M testing"
    ana_energy = (q1 * q2) / r

    # Add particles to system
    system.part.add(pos=pos1, q=q1)
    system.part.add(pos=pos2, q=q2)

    # Initialize P3M
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # --- Calculation ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    # Based on your comments: 
    # Neutral systems ~ 1e-4 error, Non-neutral ~ 0.08 error.
    # We use a broad tolerance here to cover the non-neutral case (+1, +1 charges)
    tolerance = 0.1 

    assert elc_energy == pytest.approx(ana_energy, abs=tolerance), \
        f"ELC failed analytical comparison: ELC={elc_energy}, Analytical={ana_energy}"
    
    assert elc_energy == pytest.approx(legacy_energy, abs=1e-5), \
        f"ELC and Legacy mismatch: ELC={elc_energy}, Legacy={legacy_energy}"