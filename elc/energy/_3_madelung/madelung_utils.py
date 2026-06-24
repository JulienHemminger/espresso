import numpy as np
from elc.energy.custom_elc_energy import get_elc_energy


def run_madelung(system, ions_per_axis=8, gap_size=1, accuracy=1e-6):
    system.part.clear()
    system.box_l = [10, 10, 10]

    l_xy = min(system.box_l[0], system.box_l[1])

    spacing = l_xy / ions_per_axis
    ion_z_pos = system.box_l[2] / 2.0
    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0) ** (i + j)
            system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

    ion_count = len(system.part)
    madelung_2d_ref = -1.6155426267128247 * ion_count / (2.0 * spacing)

    elc_energy = get_elc_energy(system, gap_size, accuracy)

    tolerance = 1e3 * accuracy  # 1e-3
    assert np.abs(madelung_2d_ref - elc_energy) < tolerance
