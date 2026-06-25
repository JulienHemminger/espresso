import numpy as np
import matplotlib.pyplot as plt
import espressomd
from elc.energy._5_param_sweep_rdm_tests.custom_elc_energy_for_accuracy_convergence import get_elc_energy

import numpy as np
import matplotlib.pyplot as plt
import espressomd
from common.legacy.energy import get_legacy_energy
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import get_direct_sum_energy as get_direct_sum_energy
import numpy as np
import matplotlib.pyplot as plt
import espressomd
from common.legacy.energy import get_legacy_energy
import numpy as np


def run_madelung(system, ions_list, gap_size=1, accuracy=1e-6):
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
        MADELUNG_CONSTANT_PERFECT_2D_SHEET = 1.6155426267128247 # e.g. perfect 2D sheet of NaCl
        madelung_2d_ref = -MADELUNG_CONSTANT_PERFECT_2D_SHEET  * ion_count / (2.0 * spacing)
        madelung_refs.append(madelung_2d_ref)

        elc_energy = get_elc_energy(system, gap_size, accuracy)
        elc_energies.append(elc_energy)

        print(f"{ions_per_axis=}: diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies


# Setup
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 

ions_range = range(2, 20+1, 2) # Testing different grid sizes
refs, elcs = run_madelung(system, ions_range)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-axis: Energy
color = 'tab:blue'
ax1.set_xlabel('Ions per Axis')
ax1.set_ylabel('Energy', color=color)
ax1.scatter(ions_range, refs, color='red', label='Analytical Reference', alpha=0.7)
ax1.plot(ions_range, elcs, 'x-', color=color, label='ELC Energy', alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Secondary Y-axis: Absolute Difference
ax2 = ax1.twinx()
color = 'tab:cyan'
abs_diff = np.abs(np.array(refs) - np.array(elcs))
ax2.set_ylabel('abs(Difference)', color=color)
ax2.plot(ions_range, abs_diff, 's:', color=color, label='Difference')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')  # Often useful to see differences on a log scale
ax2.legend(loc='upper right')

plt.title('2D Madelung Energy and Absolute Difference vs Grid Size, Only for even Ion Counts, since else sytems would be non-neutral')
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
