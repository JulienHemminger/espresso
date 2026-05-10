import matplotlib.pyplot as plt
import numpy as np
from elcic.energy.custom_elcic_energy import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy

from elcic.energy.single_plate.neutral.UNORGANIZED.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

def run(system, z_pos_count, params):
    prefactor = params["prefactor"]
    pw_error = params["pw_error"]
    delta_mid_top = params["delta_mid_top"]
    delta_mid_bot = params["delta_mid_bot"]
    
    eps = 0.5
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    
    legacy_energies = []
    analytical_energies = []
    ana_energy = analytical_single_plate_2d_ewald_elcic_energy([p.pos for p in system.part.all()], params["charges"], system.box_l, prefactor, delta_mid_bot, k_max=10, n_real=10)
    custom_energies = []
    
    z_range = np.linspace(eps, params["lz"] - params["gap_size"] - eps, num=z_pos_count)
    for z in z_range:
        system.part.clear()
        for i in range(min(len(params["charges"]), len(params["positions"]))):
            pos = params["positions"][i]
            system.part.add(pos=[pos[0], pos[1], z], q=params["charges"][i])

        analytical_energies.append(ana_energy)
        legacy_energies.append(get_legacy_elc_energy(system, params["gap_size"], pw_error, delta_mid_top, delta_mid_bot))
        custom_energies.append(get_elcic_energy(
            system, params["gap_size"], pw_error, prefactor, delta_mid_bot, delta_mid_top
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
    ax1.plot(z_range, analytical_energies, label='Analytical', linestyle='-')
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

   




