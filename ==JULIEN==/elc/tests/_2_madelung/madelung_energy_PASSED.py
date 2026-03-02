import pytest
import numpy as np
import espressomd
import espressomd.electrostatics
from elc.src.get_legacy_elc import get_legacy_elc_energy
from elc.src.get_elc_energy import get_elc_energy

@pytest.fixture(scope="module")
def system():
    """Create a single system instance for the entire test module."""
    s = espressomd.System(box_l=[1.0, 1.0, 1.0])
    yield s
    # No explicit 'delete' needed usually, but we ensure it's ready for the next run
    # if you were running multiple modules.

@pytest.mark.parametrize("ions_per_axis", [8, 16, 32, 64])
def test_madelung_energy_convergence(system, ions_per_axis):
    # --- 1. System Setup (Re-configuration) ---
    box_size = 20.0
    system.part.clear()
    system.box_l = [box_size, box_size, box_size * 3]
    system.cell_system.skin = 0.4
    system.time_step = 0.01
    

    # Theoretical Madelung constant for 2D square lattice
    madelung_2d_ref = -1.6155426267128247
    accuracy_goal = 1e-6

    # --- 2. Particle Placement ---
    spacing = box_size / ions_per_axis
    ion_z_pos = system.box_l[2] / 2.0

    # Optimization: Use add_bulk for large particle counts (64x64 = 4096)
    # This is much faster than a python loop for the 64x64 case
    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0)**(i + j)
            system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

    # --- 3. Configure Solver ---
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=accuracy_goal,
        check_neutrality=False
    )

    # --- 4. Calculate Energies ---
    ion_count = len(system.part)
    
    # We pass the p3m actor and system
    legacy_energy = get_legacy_elc_energy(p3m, box_size, accuracy_goal, system)
    elc_energy = get_elc_energy(p3m, box_size, accuracy_goal, system)

    # Normalize energies per ion
    legacy_energy_per_ion = legacy_energy / ion_count * (2.0 * spacing)
    elc_energy_per_ion = float(elc_energy) / ion_count * (2.0 * spacing)

    # --- 5. Assertions ---
    tolerance = 1e-3 # Madelung convergence is slower than P3M accuracy
    
    assert np.abs(madelung_2d_ref - legacy_energy_per_ion) < tolerance
    assert np.abs(madelung_2d_ref - elc_energy_per_ion) < tolerance