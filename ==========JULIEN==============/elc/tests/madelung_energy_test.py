import espressomd
import espressomd.electrostatics
import numpy as np
import matplotlib.pyplot as plt
from elc.get_legacy_elc import get_legacy_elc_energy
from elc.get_elc_energy import get_elc_energy

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

    # 2. Loop through different spacings
    spacings = [0.25, 0.5, 1, 2, 3]


    for spacing in spacings:
        spacing = float(spacing)
        system.part.clear()

        ions_per_axis = int(box_size / spacing)
        ion_z_pos = system.box_l[2] / 2.0

        for i in range(ions_per_axis):
            for j in range(ions_per_axis):
                charge = (-1.0)**(i + j)
                system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

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


    # 5. Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(ion_counts, legacy_errors, 
            marker='s',           # Square marker
            linestyle='--',       # Dashed line
            color='teal', 
            linewidth=2, 
            markersize=8, 
            alpha=0.6,            # 60% opacity
            label='Legacy Errors')

    plt.plot(ion_counts, elc_errors, 
            marker='^',           # Triangle marker
            linestyle=':',        # Dotted line
            color='red', 
            linewidth=2, 
            markersize=9,         # Slightly larger to see 'behind' squares
            alpha=0.8,            # 80% opacity
            label='ELC Errors')
    # Formatting the plot
    plt.xscale('log')  # Log scale for Number of Ions as requested
    # Note: Since the error ranges from 10^-7 to 10^-2, a log scale for Y is also highly recommended:
    plt.yscale('log') 

    plt.xlabel('Number of Ions (Log Scale)', fontsize=12)
    plt.ylabel('Absolute Error (Madelung Energy)', fontsize=12)
    plt.title('ELC Validation: Error vs. System Size', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig('elc_madelung_validation_plot.png')
    plt.show()