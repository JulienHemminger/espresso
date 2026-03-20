import numpy as np
from src.forces.get_elc_forces import get_elc_forces
from src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d


def run_madelung(system, ions_per_axis=8, gap_size=1, accuracy=1e-6):
    l_xy = min(system.box_l[0], system.box_l[1])
    spacing = l_xy / ions_per_axis
    ion_z_pos = system.box_l[2] / 2.0

    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0) ** (i + j)
            pos = [i * spacing, j * spacing, ion_z_pos]

            # BREAK SYMMETRY: Slightly nudge one particle to create non-zero forces
            if i == 0 and j == 0:
                pos[0] += 0.05 * spacing

            system.part.add(pos=pos, q=charge)

    # Calculate forces using both methods
    legacy_forces = np.array(get_ewald_forces_2d(system, n_max=100))
    elc_forces = np.array(get_elc_forces(system, gap_size, accuracy))

    # Verify that we aren't just comparing zeros
    assert np.linalg.norm(legacy_forces) > 1e-5, "Forces are zero; test is trivial!"

    # Use a relative/absolute tolerance check
    # atol should be tuned based on your method's expected precision
    np.testing.assert_allclose(
        legacy_forces,
        elc_forces,
        atol=1e3 * accuracy,
        err_msg="ELC and Legacy forces do not match!",
    )
