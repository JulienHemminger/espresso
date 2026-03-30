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

def run(system, lx, ly, lz, gap_size, charges, positions, prefactor, pw_error, delta_mid_top, delta_mid_bot, z_pos_count, params):
    eps = 0.5
    system.part.clear()
    system.box_l = [lx, ly, lz]

    legacy_energies = []
    analytical_energies = []
    custom_energies = []

    z_range = np.linspace(eps, lz - gap_size - eps, num=z_pos_count)
    for z in z_range:
        system.part.clear()
        for i in range(min(len(charges), len(positions))):
            pos = positions[i]
            system.part.add(pos=[pos[0], pos[1], z], q=charges[i])

        legacy_energies.append(get_legacy_elc_energy(system, gap_size, pw_error, delta_mid_top, delta_mid_bot))  
        analytical_energies.append(analytical_elcic_energy(system, params))
        custom_energies.append(get_elcic_energy(
            system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
        ))

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, legacy_energies, label='Legacy ELC', marker='x', linestyle='--')
    plt.plot(z_range, analytical_energies, label='Analytical ELCIC', linestyle='-')
    plt.plot(z_range, custom_energies, label='Custom ELCIC', marker='x', linestyle=':')

    plt.text(0.95, 0.95, "\n".join([f"{k}: {v}" for k, v in params.items()]), transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
             fontsize=9, family='monospace')
    
    plt.xlabel('z-position')
    plt.ylabel('Energy')
    plt.title('Energy Comparison vs Particle Position')
    plt.legend()
    plt.grid(True)
    plt.show()

   



system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01
system.cell_system.skin = 0.4
z = 1
params = {
        "lx": 9.0,
        "ly": 12.0,
        "lz": 19.0,
        "gap_size": 15.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -0.6,
        "charges": [+1, -1],
        'pw_error': 1e-8,
        "positions": [np.array([6, 5, z]), np.array([1, 3, z])]
    }

run(system, **params, z_pos_count=32, params=params)


"""
* fix the problem
    * if i change "[np.array([7, 1, z]), np.array([4, 5, z])]": NO CHANGE
    * if i change delta_bot != -1: NO CHANGE (tried "delta_mid_bot": -0.6)

* own for-loop for every method: NO CHANGE

            


* hide the problem
    * can i cap-out using by "hiding" the 0<=z<=1 error?
        * alex digs into things, he may find out

"""