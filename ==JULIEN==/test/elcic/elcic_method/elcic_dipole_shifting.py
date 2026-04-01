import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.energy.analytical_elc_energy import get_ewald_energy_2d
from common.plotting.convergence_contribution_plot import show_convergence_contribution_plot
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs, get_elcic_energy
from elcic.energy.analytical.analytical_two_plate_elcic_energy import analytical_two_plate_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy

from elcic.energy.analytical.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

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
        analytical_energies.append(analytical_single_plate_2d_ewald_elcic_energy([p.pos for p in system.part.all()], charges, system.box_l, prefactor, delta_mid_bot, k_max=10, n_real=10))
        #analytical_energies.append(analytical_two_plate_elcic_energy(system, params))
        custom_energies.append(get_elcic_energy(
            system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
        ))
        print(f"finished computing energies for {z=}")

    # Create figure with two subplots sharing the x-axis
    # Convert your lists to numpy arrays
    legacy_energies = np.array(legacy_energies)
    analytical_energies = np.array(analytical_energies)
    custom_energies = np.array(custom_energies)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top Plot: Absolute Energy
    ax1.plot(z_range, legacy_energies, label='Legacy ELC', marker='x', linestyle='--')
    ax1.plot(z_range, analytical_energies, label='Analytical ELCIC', linestyle='-')
    ax1.plot(z_range, custom_energies, label='Custom ELCIC', marker='x', linestyle=':')

    ax1.text(0.95, 0.95, "\n".join([f"{k}: {v}" for k, v in params.items()]), 
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
             fontsize=9, family='monospace')
    
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Comparison vs Particle Position')
    ax1.legend()
    ax1.grid(True)

    # Bottom Plot: Differences (Residuals)
    ax2.plot(z_range, legacy_energies - analytical_energies, label='Legacy - Analytical', marker='o', markersize=3)
    ax2.plot(z_range, custom_energies - analytical_energies, label='Custom - Analytical', marker='s', markersize=3)
    
    ax2.set_xlabel('z-position')
    ax2.set_ylabel('Difference')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

   



system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01
system.cell_system.skin = 0.4
z = 0
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
        "positions": [np.array([2, 5, z]), np.array([8, 3, z])]
    }

run(system, **params, z_pos_count=32, params=params)
