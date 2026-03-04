import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.energy.get_elc_energy import get_elc_energy
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d


@pytest.fixture(scope="module")
def system():
    """Initialize the Espresso system once for the module."""
    s = espressomd.System(box_l=[100.0, 100.0, 10.0])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

def test_non_neutral_dipole(system):
    pos1 = [50.0, 50.0, 1.0]
    pos2 = [50.0, 50.0, 9.0]
    q1 = 1.0
    q2 = 1.0
    
    # --- Setup ---
    l_xy, l_z = 100.0, 10.0
    gap_size = 1.0
    pw_error = 1e-6
    
    system.part.clear()
    system.box_l = [l_xy, l_xy, l_z]


    # Add particles to system
    system.part.add(pos=pos1, q=q1)
    system.part.add(pos=pos2, q=q2)

    # Initialize P3M
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=pw_error, 
        check_neutrality=False
    )

    # Calculate analytical baseline: E = (q1 * q2) / r
    r = math.dist(pos1, pos2)
    assert r >= 1.0, "Particles are too close for stable P3M testing"
    ana_energy = get_ewald_energy_2d(system)
    # --- Calculation ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # --- Assertions ---
    max_error = 1e2 * pw_error

    assert elc_energy == pytest.approx(ana_energy, abs=max_error), \
        f"ELC failed analytical comparison: ELC={elc_energy}, Analytical={ana_energy}"
    
    assert elc_energy == pytest.approx(legacy_energy, abs=max_error), \
        f"ELC and Legacy mismatch: ELC={elc_energy}, Legacy={legacy_energy}"
        
        