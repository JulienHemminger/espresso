import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.energy.analytical_elc_energy import get_ewald_energy_2d
from common.plotting.convergence_contribution_plot import show_convergence_contribution_plot
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs, get_elcic_energy
from elcic.energy.analytical_elcic_energy import analytical_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy

from elcic.energy.analytical_elcic_energy import analytical_elcic_energy

def run(system, lx, ly, lz, gap_size, charges, positions, prefactor, pw_error, delta_mid_top, delta_mid_bot, z_pos_count):
    system.part.clear()
    system.box_l = [lx, ly, lz]
    z_range = np.linspace(0, lz - gap_size - 1e-3, num=z_pos_count)

    legacy_energies = []
    analytical_energies = []
    custom_energies = []

    for z in z_range:
        system.part.clear()
        for  i in range(min(len(charges), len(positions))):
            pos = positions[i]
            pos[2] = z
            system.part.add(pos=pos, q=charges[i])
        

        legacy_energies.append(get_legacy_elc_energy(system, gap_size, pw_error, delta_mid_top, delta_mid_bot))
        analytical_energies.append(analytical_elcic_energy(system, params))
        custom_energies.append(get_elcic_energy(
            system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
        ))
    print([float(f) for f in analytical_energies])

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, legacy_energies, label='Legacy ELC', marker='x', linestyle='--')
    plt.plot(z_range, analytical_energies, label='Analytical ELCIC', linestyle='-')
    plt.plot(z_range, custom_energies, label='Custom ELCIC', marker='x', linestyle=':')
    
    plt.xlabel('z-position')
    plt.ylabel('Energy')
    plt.title('Energy Comparison vs Particle Position')
    plt.legend()
    plt.grid(True)
    plt.show()

   



system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01
system.cell_system.skin = 0.4

params = {
        "lx": 9.0,
        "ly": 12.0,
        "lz": 19.0,
        "gap_size": 15.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+1, -1],
        'pw_error': 1e-8,
    }
params["positions"] = [np.array([7, 1, 3]), np.array([4, 5, 2])]

run(system, **params, z_pos_count=8)