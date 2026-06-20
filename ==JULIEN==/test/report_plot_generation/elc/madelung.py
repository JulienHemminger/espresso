import numpy as np
import matplotlib.pyplot as plt
import espressomd
from elc.energy.custom_elc_energy import get_elc_energy

import numpy as np
import matplotlib.pyplot as plt
import espressomd
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elc.energy.analytical.large_box_direct_sum import get_direct_sum_energy as get_direct_sum_energy
import numpy as np
import matplotlib.pyplot as plt
import espressomd
from src.elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
from elc.energy.custom_elc_energy import get_elc_energy


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
        madelung_2d_ref = -1.6155426267128247 * ion_count / (2.0 * spacing)
        madelung_refs.append(madelung_2d_ref)

        elc_energy = get_elc_energy(system, gap_size, accuracy)
        elc_energies.append(elc_energy)

        print(f"{madelung_2d_ref=}, {elc_energy=}, diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies

"""
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# TODO create plot
# xaxis: ions_per_axis
# yaxis: madelung_2d_ref, elc_energy
run_madelung(system, ions_list=[8])

"""

# Setup
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 

ions_range = range(3, 10) # Testing different grid sizes
refs, elcs = run_madelung(system, ions_range)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-axis: Energy
color = 'tab:blue'
ax1.set_xlabel('Ions per Axis')
ax1.set_ylabel('Energy', color=color)
ax1.plot(ions_range, refs, 'o--', color=color, label='Analytical Reference', alpha=0.7)
ax1.plot(ions_range, elcs, 'x-', color=color, label='ELC Energy', alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Secondary Y-axis: Absolute Difference
ax2 = ax1.twinx()
color = 'tab:red'
abs_diff = np.abs(np.array(refs) - np.array(elcs))
ax2.set_ylabel('abs(Difference)', color=color)
ax2.plot(ions_range, abs_diff, 's:', color=color, label='Difference')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')  # Often useful to see differences on a log scale
ax2.legend(loc='upper right')

plt.title('2D Madelung Energy and Absolute Difference vs Grid Size')
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# fix reference_energy for ions_per_axis odd (1, 3, 5, ...)