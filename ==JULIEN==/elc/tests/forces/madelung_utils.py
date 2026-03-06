from elc.src.common.set_utils import are_sets_equal
from elc.src.forces.get_elc_forces import get_elc_forces
from elc.src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d


def run_madelung(system, ions_per_axis=8, gap_size=1, accuracy=1e-6):
    l_xy = min(system.box_l[0], system.box_l[1])

    spacing = l_xy / ions_per_axis
    ion_z_pos = system.box_l[2] / 2.0
    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0) ** (i + j)
            system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

    ion_count = len(system.part)
    # madelung_2d_ref = -1.6155426267128247 * ion_count / (2.0 * spacing)

    legacy_forces = get_ewald_forces_2d(system, n_max=100)
    elc_forces = get_elc_forces(system, gap_size, accuracy)

    tolerance = 1e3 * accuracy  # 1e-3
    assert are_sets_equal(legacy_forces, elc_forces, tolerance)
    # assert np.abs(madelung_2d_ref - legacy_energy) < tolerance
    # assert np.abs(madelung_2d_ref - elc_energy) < tolerance
