import numpy as np
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import (
    get_direct_sum_energy as get_direct_sum_energy,
)
from elc.energy._1_big_box_neutral_dipole.custom1 import get_elc_energy_contribs

def get_madelung_energy(system, ions_list, gap_size=1, accuracy=1e-8):
    madelung_refs = []
    elc_energies = []

    for ions_per_axis in ions_list:
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
        MADELUNG_CONSTANT_PERFECT_2D_SHEET = (
            1.6155426267128247  # e.g. perfect 2D sheet of NaCl
        )
        madelung_2d_ref = (
            -MADELUNG_CONSTANT_PERFECT_2D_SHEET * ion_count / (2.0 * spacing)
        )
        madelung_refs.append(madelung_2d_ref)


        E_3d, E_dipole, E_recip, E_nonneutral = get_elc_energy_contribs(gap_size=gap_size, pw_error=accuracy, system=system, prefactor=1.0)
        elc_energy = E_3d + E_dipole + E_recip + E_nonneutral
        elc_energies.append(elc_energy)

        print(f"{ions_per_axis=}: diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies
