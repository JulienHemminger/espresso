import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

import os
from datetime import datetime

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def run_lerp_plot(system, start_params, end_params, get_custom_energy, get_analytical_energy=None, get_legacy_energy=None, steps=20):
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
        
        # 1. Legacy ELC (Optional)
        if get_legacy_energy is not None:
            results["legacy"].append(get_legacy_energy(system, lerp_params))
            
        # 2. Analytical (Optional)
        if get_analytical_energy is not None:
            results["analytical"].append(get_analytical_energy(lerp_params))
            
        # 3. Custom ELCIC (Always computed)
        results["custom"].append(get_custom_energy(system, lerp_params))

    # --- Console Output Summary ---
    print("\n" + "="*80)
    print("LERP EVALUATION SUMMARY")
    print("="*80)

    for i, t in enumerate(t_values):
        current_params = {
            "lx":            lerp(start_params["lx"],            end_params["lx"],            t),
            "ly":            lerp(start_params["ly"],            end_params["ly"],            t),
            "lz":            lerp(start_params["lz"],            end_params["lz"],            t),
            "gap_size":      lerp(start_params["gap_size"],      end_params["gap_size"],      t),
            "delta_mid_top": lerp(start_params["delta_mid_top"], end_params["delta_mid_top"], t),
            "delta_mid_bot": lerp(start_params["delta_mid_bot"], end_params["delta_mid_bot"], t),
            "pw_error":      lerp(start_params["pw_error"],      end_params["pw_error"],      t),
            "prefactor":     lerp(start_params["prefactor"],     end_params["prefactor"],     t)
        }

        param_str = ", ".join([f"{k}: {v}" if isinstance(v, (float, int)) else f"{k}: {v}" 
                              for k, v in current_params.items()])
        
        print(f"Step {i+1}/{steps} (t={t:.3f})")
        print(f"parameters = {{{param_str}}}")
        
        e_cus = results["custom"][i]
        
        # Print logs dynamically depending on what parameters were passed
        if get_analytical_energy is not None:
            e_ana = results["analytical"][i]
            print(f"analytical_energy = {e_ana:.8f}")
            print(f"custom_energy     = {e_cus:.8f} (error={abs(e_cus - e_ana):.2e} compared to analytical)")
            
            if get_legacy_energy is not None:
                e_leg = results["legacy"][i]
                print(f"legacy_energy     = {e_leg:.8f} (error={abs(e_leg - e_ana):.2e} compared to analytical)")
        else:
            print(f"custom_energy     = {e_cus:.8f}")
            if get_legacy_energy is not None:
                e_leg = results["legacy"][i]
                print(f"legacy_energy     = {e_leg:.8f} (diff={abs(e_cus - e_leg):.2e} compared to custom)")
                
        print("-" * 80)

    # --- Label Construction ---
    label_lines = ["**Parameters**"]
    for key in start_params.keys():
        val_start = start_params[key]
        val_end = end_params[key]
        
        if np.array_equal(val_start, val_end):
            label_lines.append(f"{key}: {val_start}")
        else:
            if isinstance(val_start, (list, np.ndarray)):
                label_lines.append(f"{key}: [Changed]")
            else:
                label_lines.append(f"{key}: {val_start} → {val_end}")

    param_text = "\n".join(label_lines)

    # --- Plotting (Two Subplots) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Cache arrays for easier access
    y_cus = np.array(results["custom"])
    y_ana = np.array(results["analytical"]) if get_analytical_energy is not None else None
    y_leg = np.array(results["legacy"]) if get_legacy_energy is not None else None

    # --- 1. Top Subplot: Absolute Energy Values ---
    ax1.plot(t_values, y_cus, label="Custom Energy", color="green", lw=2)
    if y_ana is not None:
        ax1.plot(t_values, y_ana, label="Analytical Energy", color="black", ls="--", lw=2)
    if y_leg is not None:
        ax1.plot(t_values, y_leg, label="Legacy Energy", color="orange", ls=":", lw=2)
        
    ax1.set_ylabel("Total Energy")
    ax1.set_title("Evaluation over Lerp Axis")
    #ax1.set_yscale("log")
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # --- 2. Bottom Subplot: Differences / Errors ---
    if y_ana is not None:
        ax2.plot(t_values, np.abs(y_cus - y_ana), label="Error: Custom - Analytical", color="blue", lw=2)
        if y_leg is not None:
            ax2.plot(t_values, np.abs(y_leg - y_ana), label="Error: Legacy - Analytical", color="red", lw=2)
        ax2.set_ylabel("Absolute Energy Error")
    elif y_leg is not None:
        ax2.plot(t_values, np.abs(y_cus - y_leg), label="Difference: Custom - Legacy", color="purple", lw=2)
        ax2.set_ylabel("Absolute Energy Difference")
    else:
        # Fallback if only custom energy exists
        ax2.plot(t_values, np.zeros_like(t_values), label="No comparison baseline available", color="gray", ls="-.")
        ax2.set_ylabel("Difference")

    ax2.set_xlabel(r"Interpolation Parameter $t$")
    ax2.legend(loc='upper left')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    
    # Only use log scale if we have a baseline to compare against
    if get_analytical_energy is not None or get_legacy_energy is not None:
        ax2.yscale = "log"  # Just to track the intent, correct method below:
        ax2.set_yscale("log")

    # --- Parameter Box Placement ---
    # Placed relative to the entire figure so it centers neatly beside both subplots
    fig.text(1.02, 0.5, param_text, verticalalignment='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             transform=fig.transFigure)

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    
    # Save logic
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lerp2d_{timestamp}.png"
    save_path = os.path.join("/home/main", filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()