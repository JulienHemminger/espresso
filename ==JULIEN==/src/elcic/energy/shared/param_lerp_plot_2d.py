import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elcic.energy.single_plate.neutral.metallic.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy
from elcic.energy.single_plate.neutral.metallic.analytical import (
    analytical_single_plate_2d_ewald_elcic_energy,
)
import os
from datetime import datetime

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def run_lerp_plot(system, start_params, end_params, steps=20):
    assert len(start_params["positions"]) == len(start_params["charges"])
    assert len(end_params["positions"]) == len(end_params["charges"])
    assert len(start_params["charges"]) == len(end_params["charges"]), (
        "To lerp, start_params and end_params need the same number of particles"
    )

    t_values = np.linspace(0, 1, steps)
    results = {"legacy": [], "analytical": [], "custom": []}

    for t in t_values:
        system.electrostatics.clear() # NEED this when resetting box_l

        lerp_params = {
            "lx":            lerp(start_params["lx"],            end_params["lx"],            t),
            "ly":            lerp(start_params["ly"],            end_params["ly"],            t),
            "lz":            lerp(start_params["lz"],            end_params["lz"],            t),
            "gap_size":      lerp(start_params["gap_size"],      end_params["gap_size"],      t),
            "delta_mid_top": lerp(start_params["delta_mid_top"], end_params["delta_mid_top"], t),
            "delta_mid_bot": lerp(start_params["delta_mid_bot"], end_params["delta_mid_bot"], t),
            "pw_error":      lerp(start_params["pw_error"],      end_params["pw_error"],      t),
            "prefactor":     lerp(start_params["prefactor"],     end_params["prefactor"],     t),
            "positions": [
                lerp(np.array(p_start), np.array(p_end), t) 
                for p_start, p_end in zip(start_params["positions"], end_params["positions"])
            ],
            "charges": [
                lerp(q_start, q_end, t) 
                for q_start, q_end in zip(start_params["charges"], end_params["charges"])
            ]
        }

        # Update the EspressoMD system state
        system.part.clear()
        system.box_l = [lerp_params["lx"], lerp_params["ly"], lerp_params["lz"]]
        
        for i in range(len(lerp_params["charges"])):
            system.part.add(pos=lerp_params["positions"][i], q=lerp_params["charges"][i])

        # --- Energy Calculations ---
        
        # 1. Legacy ELC (Updated to take current_params dict)
        results["legacy"].append(
            get_legacy_elc_energy(system, lerp_params)
        )
        # 2. Analytical (Takes current_params dict)
        results["analytical"].append(
            analytical_single_plate_2d_ewald_elcic_energy(lerp_params)
        )
        # 3. Custom ELCIC (Updated to take current_params dict)
        results["custom"].append(
            get_elcic_energy(system, lerp_params)
        )
    print(results["analytical"])

    # Plotting according to sketch: Error vs t
    plt.figure(figsize=(10, 6))

    # Calculating absolute error relative to analytical for the 'Error' plot
    err_legacy = np.abs(np.array(results["legacy"]) - np.array(results["analytical"]))
    err_custom = np.abs(np.array(results["custom"]) - np.array(results["analytical"]))

    plt.plot(t_values, err_legacy, label="Error: ana-legacy", color="red", lw=2)
    plt.plot(t_values, err_custom, label="Error: ana-custom", color="blue", lw=2)

    plt.xlabel(r"Interpolation Parameter $t$ (Start $\to$ End)")
    plt.ylabel("Absolute Energy Error")
    plt.title("Evaluation of Multiple Lerp Axes")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.yscale("log")
    
    # Generate filename with current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lerp2d_{timestamp}.png"
    save_path = os.path.join("/home/main", filename)

    # Save and close to free up memory
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()
