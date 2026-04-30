import matplotlib.pyplot as plt
import numpy as np
import espressomd
import mpld3
import time
from pathlib import Path
from elcic.energy.custom_elcic_energy import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy
from elcic.energy.analytical.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

def run_z_sweep(system, z_steps, params):
    # Fixed parameters from your setup
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    gap_size = params["gap_size"]
    prefactor = params["prefactor"]
    pw_error = params["pw_error"]
    delta_top = params["delta_mid_top"]
    delta_bot = params["delta_mid_bot"]
    charges = params["charges"]
    
    # Initialize positions (Particle 0 z will be overwritten in loop)
    # Using your start_pos as the base
    base_positions = [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])]
    
    # Define Z range for Particle 0
    # Ensuring we stay within the slab (avoiding the gap and small epsilon offset)
    eps = 5e-3
    z_min = eps
    z_max = lz - gap_size - eps
    z_values = np.linspace(z_min, z_max, num=z_steps)

    legacy_energies = []
    custom_energies = []
    analytical_energies = []

    system.box_l = [lx, ly, lz]

    for z in z_values:
        # Update Particle 0's Z position
        current_pos = [p.copy() for p in base_positions]
        current_pos[0][2] = z 
        #current_pos[1][2] = z 
        
        # 1. Update System particles
        system.part.clear()
        for i in range(len(charges)):
            system.part.add(pos=current_pos[i], q=charges[i])

        # 2. Compute Analytical (Fixed k_max and n_real for comparison)
        ana = analytical_single_plate_2d_ewald_elcic_energy(

            positions=[p.pos for p in system.part.all()], 
            charges=charges, box_l=system.box_l, prefactor=prefactor, delta_mid_bot=delta_bot, 
            k_max=10, n_real=10
        )
        analytical_energies.append(ana)

        # 3. Compute Legacy ELC
        legacy_energies.append(
            get_legacy_elc_energy(
                system=system, gap_size=gap_size, pw_error=pw_error, delta_mid_top=delta_top, delta_mid_bot=delta_bot)
        )

        # 4. Compute Custom ELCIC
        custom_energies.append(
            get_elcic_energy(system=system, gap_size=gap_size, pw_error=pw_error, prefactor=prefactor, delta_mid_bot=delta_bot, delta_mid_top=delta_top)
        )
        
        print(f"Computed z={z:.3f}")

    # --- Plotting ---
    legacy_energies = np.array(legacy_energies)
    analytical_energies = np.array(analytical_energies)
    custom_energies = np.array(custom_energies)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: Absolute
    ax1.plot(z_values, legacy_energies, label='Legacy ELC', marker='x', linestyle='--')
    ax1.plot(z_values, analytical_energies, label='Analytical', linestyle='-', color='black', alpha=0.6)
    ax1.plot(z_values, custom_energies, label='Custom ELCIC', marker='+', linestyle=':')
    
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Sweep: Particle 1 Z-position (q={charges})')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Bottom: Residuals
    ax2.plot(z_values, legacy_energies - analytical_energies, label='Legacy - Analytical', marker='o', markersize=3)
    ax2.plot(z_values, custom_energies - analytical_energies, label='Custom - Analytical', marker='s', markersize=3)
    
    ax2.set_xlabel('Particle 1 z-coordinate')
    ax2.set_ylabel('Difference')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    output_dir = Path("/home/main")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_base = output_dir / f"zshifting_{timestamp}"
    html_path = f"{filename_base}.html"
    mpld3.save_html(fig, html_path)
    print(f"Interactive plot saved to: {html_path}")


# --- Execution ---
if __name__ == "__main__":
    system = espressomd.System(box_l=[1, 1, 1])
    system.time_step = 0.01

    # Your specific parameters
    params = {
        "lx": 22.553239340502152,
        "ly": 41.839080168737745,
        "lz": 35.51793215353896,
        "gap_size": 32.234687817568656,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": 1e-8,
        "charges": [+1, -1],
    }


    run_z_sweep(system, z_steps=36, params=params)
