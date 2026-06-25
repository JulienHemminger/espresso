import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import (
    get_ewald_energy_2d
)

def run(system, lx, ly, lz, gap_size, charges, positions, z_pos_count):
    system.part.clear()
    system.box_l = [lx, ly, lz]
    z_range = np.linspace(0, lz - gap_size - 1e-3, num=z_pos_count)
    ewald_energies = []

    for z in z_range:
        system.part.clear()
        for  i in range(min(len(charges), len(positions))):
            pos = positions[i]
            pos[2] = z
            system.part.add(pos=pos, q=charges[i])
        

        e_ewald = get_ewald_energy_2d(system)
        ewald_energies.append(e_ewald)


    plt.figure(figsize=(10, 6))

    plt.plot(
        z_range,
        ewald_energies,
        "-",
        color="tab:red",
        linewidth=2,
        label="Ewald 2D Energy ($n_{max}=100$)",
    )

    plt.xlabel("Dipole Height ($z$)")
    plt.ylabel("Interaction Energy")
    plt.title("2D Ewald Energy vs. Vertical Position (z)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()



system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01
system.cell_system.skin = 0.4
run(system, 10, 10, 3, 1, np.array([+2, -1]), np.array([[3.0, 5.0, 0], [7.0, 5.0, 0]]), 10)