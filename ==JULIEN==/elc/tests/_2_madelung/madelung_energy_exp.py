import espressomd
import espressomd.electrostatics
import numpy as np
import matplotlib.pyplot as plt
from elc.src.get_legacy_elc import get_legacy_elc_energy
from elc.src.get_elc_energy import get_elc_energy

def madelung_energy_test():
    # 1. System Setup
    box_size = 20.0
    system = espressomd.System(box_l=[box_size, box_size, box_size * 3])
    system.cell_system.skin = 0.4
    system.time_step = 0.01

    # Theoretical Madelung constant for 2D square lattice
    madelung_2d_ref = -1.6155426267128247

    # Data storage for plotting
    ion_counts = []
    legacy_errors = []
    elc_errors = []

    ions_per_axis_list = [8, 16, 32, 64] 

    for ions_per_axis in ions_per_axis_list:
        # Calculate spacing based on the desired number of ions
        spacing = box_size / ions_per_axis
        
        system.part.clear()
        ion_z_pos = system.box_l[2] / 2.0

        for i in range(ions_per_axis):
            for j in range(ions_per_axis):
                charge = (-1.0)**(i + j)
                system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

        # 3. Configure the Solvers & 4. Calculate Energy
        # (Your existing P3M and error calculation logic remains the same)
        # Note: Use 'ions_per_axis**2' for total ion_count
        ion_count = ions_per_axis**2

        # 3. Configure the Solvers
        p3m = espressomd.electrostatics.P3M(
            prefactor=1.0, 
            accuracy=1e-6, 
        )
        legacy_energy = get_legacy_elc_energy(p3m, box_size, 1e-6, system)
        
        # 4. Calculate Energy and Error
        ion_count = len(system.part)
        legacy_energy_per_ion = legacy_energy / ion_count * (2.0 * spacing)
        legacy_error = np.abs(madelung_2d_ref - legacy_energy_per_ion)

        elc_energy = get_elc_energy(p3m, box_size, 1e-6, system)
        elc_energy_per_ion = elc_energy / ion_count * (2.0 * spacing)
        elc_error = np.abs(madelung_2d_ref - elc_energy_per_ion)
        # Store results
        ion_counts.append(ion_count)
        legacy_errors.append(legacy_error)
        elc_errors.append(elc_error)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Added labels, distinct markers ('o' and 's'), and transparency (alpha)
    ax.scatter(ion_counts, elc_errors, color='#2980b9', s=30, marker='o', alpha=0.3, label='ELC')
    ax.scatter(ion_counts, legacy_errors, color="#ff0000", s=30, marker='s', alpha=0.3, label='Legacy')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_ylabel(r'Diff ($\Delta E$)')
    ax.set_xlabel(r'Number of Ions (Log Scale)')
    ax.set_title('Residuals of Energy Computation', fontweight='bold', pad=10)

    # Display the legend to show the labels
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=11, frameon=True)

    # Styling
    plt.grid(True, which="both", ls="-", alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()