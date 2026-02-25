import matplotlib.pyplot as plt
import numpy as np
import math
from common.get_positions import get_rdm_constrained_point_pairs, get_rdm_point, get_rdm_constrained_points
from elc.src.get_elc_energy import get_elc_energy

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy
import numpy as np
import matplotlib.pyplot as plt
import espressomd
from common.get_charges import get_rdm_charges_neutral, get_rdm_charges


import numpy as np
import espressomd
import matplotlib.pyplot as plt
import random

def dipole_rdm_pos_energy_non_square_test():

    l_x = 100.0 # keep l_xy <= 200
    l_y = 50.0
    l_z = 10.0

    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    # Parameters for both methods + Initialize P3M deterministically
    pw_error = 1e-6
    gap_size = 1.0
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, check_neutrality=False)


    # Lists to store data for plotting
    particle_counts = [2]
    delta_energies = []

    for point_count in particle_counts:
        system.part.clear()


        rs = get_rdm_constrained_points(l_x, l_y, l_z-gap_size-1e-3, point_count)
        qs = 99 * [+1.0, -1.0]  # get_rdm_charges(point_count)
        for i in range(min(len(rs), len(qs))):
            system.part.add(pos=rs[i], q=qs[i])
            print(f"Add particle ({rs[i]}, {qs[i]})")


        # Calculate energies
        legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
        elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

        delta_energies.append(legacy_energy - elc_energy)

    particle_counts = np.array(particle_counts)
    delta_energies = np.array(delta_energies)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # --- Residual Plot ---
    # Added labels, distinct markers ('o' and 's'), and transparency (alpha)
    ax.scatter(particle_counts, delta_energies, color="#ff0000", s=30, marker='o', alpha=0.6, label='Legacy - ELC')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_ylabel(r'Diff ($\Delta E$)')
    ax.set_xlabel(r'Particle count (n)')
    ax.set_title('Residuals of Energy Computation', fontweight='bold', pad=10)

    # Display the legend to show the labels
    ax.legend(frameon=False)

    # Styling
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()