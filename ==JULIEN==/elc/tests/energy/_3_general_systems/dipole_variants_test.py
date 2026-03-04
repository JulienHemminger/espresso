import pytest
import numpy as np
import espressomd
import espressomd.electrostatics
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
from elc.src.energy.get_elc_energy import get_elc_energy

@pytest.fixture(scope="module")
def system():
    """Maintain a single Espresso system instance for the module."""
    s = espressomd.System(box_l=[100, 100, 10])
    s.time_step = 0.01
    s.cell_system.skin = 0.4
    yield s

def run_comparison(l_xy, l_z, gap, system):
    """Helper to reset system and return both energy calculations."""
    accuracy = 1e-6
    params = {'accuracy': accuracy, 'prefactor': 1.0}
    
    system.part.clear()
    system.box_l = [l_xy, l_xy, l_z]
    
    # Place a simple dipole in the center of the slab
    z_mid = (l_z - gap) / 2.0
    system.part.add(pos=[l_xy/2, l_xy/2, z_mid - 0.5], q=+1.0)
    system.part.add(pos=[l_xy/2, l_xy/2, z_mid + 0.5], q=-1.0)

    p3m = espressomd.electrostatics.P3M(**params)
    
    legacy_e = get_legacy_elc_energy(p3m, gap, accuracy, system)
    newer_e = get_elc_energy(p3m, gap, accuracy, system)
    
    return float(legacy_e), float(newer_e)

# --- Parametrized Tests ---

@pytest.mark.parametrize("gap", np.linspace(1.0, 5.0, 3))
def test_dipole_varying_gap(system, gap):
    legacy, newer = run_comparison(100.0, 10.0, gap, system)
    assert newer == pytest.approx(legacy, abs=1e-4), \
        f"Mismatch at gap_size={gap}: New={newer}, Legacy={legacy}"

@pytest.mark.parametrize("l_xy", np.linspace(50.0, 200.0, 3))
def test_dipole_varying_l_xy(system, l_xy):
    legacy, newer = run_comparison(l_xy, 10.0, 2.0, system)
    assert newer == pytest.approx(legacy, abs=1e-4), \
        f"Mismatch at l_xy={l_xy}: New={newer}, Legacy={legacy}"

@pytest.mark.parametrize("l_z", np.linspace(8.0, 20.0, 3))
def test_dipole_varying_l_z(system, l_z):
    legacy, newer = run_comparison(100.0, l_z, 2.0, system)
    assert newer == pytest.approx(legacy, abs=1e-4), \
        f"Mismatch at l_z={l_z}: New={newer}, Legacy={legacy}"