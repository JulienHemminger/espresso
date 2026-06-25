import numpy as np
import matplotlib.pyplot as plt
from elc.force.get_custom_elc_forces import get_elc_forces
from elc.force._2_small_box_neutral.reference_method.get_ewald_forces import get_ewald_forces_2d
import espressomd

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
    reference_forces = np.array(get_ewald_forces_2d(system, n_max=100))
    elc_forces = np.array(get_elc_forces(system, gap_size, accuracy))

    # Calculate magnitude of forces for each particle for plotting
    ref_magnitudes = np.linalg.norm(reference_forces, axis=1)
    elc_magnitudes = np.linalg.norm(elc_forces, axis=1)
    particle_indices = np.arange(len(ref_magnitudes))

    # --- Plotting Configuration ---
    plt.figure(figsize=(10, 6))
    
    # Plot ground truth reference forces
    plt.plot(particle_indices, ref_magnitudes, 'o-', label='Reference (Ewald 2D)', markersize=6, alpha=0.8)
    
    # Plot ELC forces
    plt.plot(particle_indices, elc_magnitudes, 'x--', label='ELC Forces', markersize=6, alpha=0.8)
    
    plt.title('Comparison of Particle Force Magnitudes: Ewald 2D vs. ELC')
    plt.xlabel('Particle Index')
    plt.ylabel('Force Magnitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4
run_madelung(system, ions_per_axis=8)