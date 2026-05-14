import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

import os
from datetime import datetime

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def run_lerp_plot(system, start_params, end_params, get_analytical_energy, get_legacy_energy, get_custom_energy, steps=20):
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
            get_legacy_energy(system, lerp_params)
        )
        # 2. Analytical (Takes current_params dict)
        results["analytical"].append(
            get_analytical_energy(lerp_params)
        )
        # 3. Custom ELCIC (Updated to take current_params dict)
        results["custom"].append(
            get_custom_energy(system, lerp_params)
        )

    # --- Console Output Summary ---
    print("\n" + "="*80)
    print("LERP EVALUATION SUMMARY")
    print("="*80)

    for i, t in enumerate(t_values):
        # Re-calculate the specific lerp_params for this t to print them
        # (This mirrors the logic in the loop above)
        current_params = {
            "lx":            lerp(start_params["lx"],            end_params["lx"],            t),
            "ly":            lerp(start_params["ly"],            end_params["ly"],            t),
            "lz":            lerp(start_params["lz"],            end_params["lz"],            t),
            "gap_size":      lerp(start_params["gap_size"],      end_params["gap_size"],      t),
           
            "pw_error":      lerp(start_params["pw_error"],      end_params["pw_error"],      t),
            "prefactor":     lerp(start_params["prefactor"],     end_params["prefactor"],     t)
        }

        e_ana = results["analytical"][i]
        e_leg = results["legacy"][i]
        e_cus = results["custom"][i]
        
        err_leg = abs(e_leg - e_ana)
        err_cus = abs(e_cus - e_ana)

        # Print the parameter dictionary for this step
        param_str = ", ".join([f"{k}: {v}" if isinstance(v, (float, int)) else f"{k}: {v}" 
                              for k, v in current_params.items()])
        
        print(f"Step {i+1}/{steps} (t={t:.3f})")
        print(f"parameters = {{{param_str}}}")
        print(f"analytical_energy = {e_ana:.8f}")
        print(f"legacy_energy     = {e_leg:.8f} (error={err_leg:.2e} compared to analytical)")
        print(f"custom_energy     = {e_cus:.8f} (error={err_cus:.2e} compared to analytical)")
        print("-" * 80)

    # --- Label Construction ---
    label_lines = ["**Parameters**"]
    
    for key in start_params.keys():
        val_start = start_params[key]
        val_end = end_params[key]
        
        # Check if values are identical (handling both scalars and numpy arrays/lists)
        if np.array_equal(val_start, val_end):
            label_lines.append(f"{key}: {val_start}")
        else:
            # Special formatting for lists/arrays to keep the label clean
            if isinstance(val_start, (list, np.ndarray)):
                label_lines.append(f"{key}: [Changed]")
            else:
                label_lines.append(f"{key}: {val_start} → {val_end}")

    param_text = "\n".join(label_lines)

    # --- Plotting ---
    plt.figure(figsize=(12, 7)) # Increased width for the text box

    err_legacy = np.abs(np.array(results["legacy"]) - np.array(results["analytical"]))
    err_custom = np.abs(np.array(results["custom"]) - np.array(results["analytical"]))

    plt.plot(t_values, err_legacy, label="Error: Analytical-Legacy", color="red", lw=2)
    plt.plot(t_values, err_custom, label="Error: Analytical-Custom", color="blue", lw=2)

    # Add the text box to the right of the plot
    plt.text(1.02, 0.5, param_text, transform=plt.gca().transAxes, 
             verticalalignment='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.xlabel(r"Interpolation Parameter $t$")
    plt.ylabel("Absolute Energy Error")
    plt.title("Evaluation over Lerp Axis")
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.yscale("log")
    
    # Adjust layout to make room for the label on the right
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    
    # Save logic...
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lerp2d_{timestamp}.png"
    save_path = os.path.join("/home/main", filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()
