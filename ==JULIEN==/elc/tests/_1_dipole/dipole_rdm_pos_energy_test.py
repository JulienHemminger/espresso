import matplotlib.pyplot as plt
import numpy as np
import math
from common.generate_constrained_position_pairs import generate_constrained_pairs
from elc.src.get_elc_energy import get_elc_energy

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy


def dipole_rdm_pos_energy_test(test_count = 2):
    l_xy = 100.0 # keep l_xy <= 200
    l_z = 10.0

    system = espressomd.System(box_l=[l_xy, l_xy, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    # Parameters for both methods + Initialize P3M deterministically
    pw_error = 1e-6
    gap_size = 1.0
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)



    # Lists to store data for plotting
    r_values = []
    legacy_energies = []
    elc_energies = []
    ana_energies = []

    for pos1, pos2 in generate_constrained_pairs(test_count, box_size=min(l_xy, l_z-gap_size-1e-3)):
        r = math.dist(pos1, pos2)
        assert r >= 1
        
        system.part.clear() # remove all particles
        system.part.add(pos=pos1, q=+1.0)
        system.part.add(pos=pos2, q=-1.0)

        # Calculate energies
        ana_energy = -1.0 / r
        legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
        elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

        # Append to lists
        r_values.append(r)
        legacy_energies.append(legacy_energy)
        elc_energies.append(elc_energy)
        ana_energies.append(ana_energy)

    # Convert to numpy arrays and sort by r to ensure the lines are drawn correctly
    sort_idx = np.argsort(r_values)
    r_values = np.array(r_values)[sort_idx]
    legacy_energies = np.array(legacy_energies)[sort_idx]
    elc_energies = np.array(elc_energies)[sort_idx]
    ana_energies = np.array(ana_energies)[sort_idx]

    # Create a figure with two rows, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})

    # --- Main Plot (Top) ---
    ax1.plot(r_values, ana_energies, label=r'Analytical Reference', 
            color='#2c3e50', linewidth=2, zorder=1)
    ax1.scatter(r_values, legacy_energies, label='Legacy Energy', color="#15ff00", s=30, edgecolor='white', linewidth=0.5, zorder=2)

    ax1.scatter(r_values, elc_energies, label='ELC Energy', 
                color="#e2402e", s=30, edgecolor='white', linewidth=0.5, zorder=2)

    ax1.set_ylabel(r'Total Energy $E(r)$')
    ax1.set_title('Validation of Energy Computation: Dipole System', fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=True)

    # --- Residual Plot (Bottom) ---
    residuals = np.array(elc_energies) - np.array(ana_energies)
    ax2.scatter(r_values, residuals, color='#2980b9', s=20, marker='D')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_ylabel(r'Diff ($\Delta E$)')
    ax2.set_xlabel(r'Inter-particle distance ($r$)')

    # Clean up styling for both
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()