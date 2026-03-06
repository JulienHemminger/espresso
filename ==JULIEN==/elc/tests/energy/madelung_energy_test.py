import espressomd
import espressomd.electrostatics
import numpy as np
from elc.src.energy.get_elc_energy import get_elc_energy
from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d


def test_madelung():
    system = espressomd.System(box_l=[20, 20, 60])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    run_madelung(system, ions_per_axis=8)


def run_madelung(system, ions_per_axis=8, gap_size=1, accuracy=1e-6):
    l_xy = min(system.box_l[0], system.box_l[1])

    spacing = l_xy / ions_per_axis
    ion_z_pos = system.box_l[2] / 2.0
    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0) ** (i + j)
            system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

    ion_count = len(system.part)
    madelung_2d_ref = -1.6155426267128247 * ion_count / (2.0 * spacing)

    legacy_energy = get_ewald_energy_2d(system)
    elc_energy = get_elc_energy(system, gap_size, accuracy)

    tolerance = 1e3 * accuracy  # 1e-3
    assert np.abs(madelung_2d_ref - legacy_energy) < tolerance
    assert np.abs(madelung_2d_ref - elc_energy) < tolerance
