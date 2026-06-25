import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from common.plotting.utils.plot_saving import save_plot_with_timestamp

def normalize_to_dict(val, default_key="energy"):
    """Helper to ensure energy results are always dictionaries."""
    if isinstance(val, dict):
        return val
    return {default_key: val}

# Configuration for easier maintenance
NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor"]
OPTIONAL_KEYS = ["delta_mid_top", "delta_mid_bot"]

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b

def lerp_dict(start_params, end_params, t):
    """Modular interpolation of parameters."""
    lerp_params = {key: lerp(start_params[key], end_params[key], t) for key in NESSECARY_KEYS}
    
    for key in OPTIONAL_KEYS:
        if key in start_params and key in end_params:
            lerp_params[key] = lerp(start_params[key], end_params[key], t)
            
    lerp_params["positions"] = [
        lerp(np.array(p_start), np.array(p_end), t)
        for p_start, p_end in zip(start_params["positions"], end_params["positions"])
    ]
    lerp_params["charges"] = [
        lerp(q_start, q_end, t)
        for q_start, q_end in zip(start_params["charges"], end_params["charges"])
    ]
    return lerp_params

def get_param_label(start_params, end_params):
    label_lines = ["**Parameters**"]
    for key in start_params.keys():
        v1, v2 = start_params[key], end_params[key]
        
        if key == "positions":
            label_lines.append("positions:")
            for i, (p1, p2) in enumerate(zip(v1, v2)):
                # Use np.array_equal to safely compare two arrays
                if np.array_equal(p1, p2):
                    label_lines.append(f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1)}]")
                else:
                    label_lines.append(f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1)}] → [{', '.join(f'{x:.1f}' for x in p2)}]")
        else:
            # For scalar parameters, direct comparison works fine
            if v1 == v2:
                label_lines.append(f"{key}: {v1}")
            else:
                label_lines.append(f"{key}: {v1} → {v2}")

    return "\n".join(label_lines)

def run_lerp_plot(system, start_params, end_params, get_custom_energy, get_legacy_energy=None, steps=20):
    t_values = np.linspace(0, 1, steps)
    results = {"legacy": [], "custom": []}

    # Run Simulation/Evaluation Loop
    for t in t_values:
        system.electrostatics.clear()
        params = lerp_dict(start_params, end_params, t)
        
        system.part.clear()
        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])


        # Capture full dictionaries
        results["custom"] = [normalize_to_dict(r) for r in results["custom"]]
        if get_legacy_energy:
            results["legacy"] = [normalize_to_dict(r) for r in results["legacy"]]

        
        for i in range(steps):

            
            t = t_values[i]
            params = lerp_dict(start_params, end_params, t)
            print(f"Parameters={params}")
            print(f"{'Name':<12} | {'Legacy':<10} | {'Custom':<10} | {'Error (abs)'}")
            print("-" * 50)
            leg_final = results["legacy"][i]
            cus_final = results["custom"][i]
            
            for custom_key in cus_final.keys():
                legacy_key = next((lk for lk in leg_final.keys() if lk.lower() == custom_key.lower()), None)
                if legacy_key:
                    val_l = float(leg_final[legacy_key])
                    val_c = float(cus_final[custom_key])
                    error = abs(val_l - val_c)
                    print(f"{custom_key:<12} | {val_l:<15.10f} | {val_c:<15.10f} | {error:<15.10f}")

    # --- Prepare Data ---
    # Convert list of dicts to a dict of clean float numpy arrays (handles np.float64 objects safely)
    custom_data = {k: np.array([float(d[k]) for d in results["custom"]], dtype=float) for k in results["custom"][0].keys()}
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if get_legacy_energy and len(results["legacy"]) > 0:
        legacy_data = {k: np.array([float(d[k]) for d in results["legacy"]], dtype=float) for k in results["legacy"][0].keys()}
        
        # Match keys case-insensitively (e.g., matching 'e_total' from custom with 'E_total' from legacy)
        for custom_key in custom_data.keys():
            legacy_key = next((lk for lk in legacy_data.keys() if lk.lower() == custom_key.lower()), None)
            
            if legacy_key:
                # Calculate absolute difference using the clean, mapped float arrays
                diff = np.abs(custom_data[custom_key] - legacy_data[legacy_key])
                ax.plot(t_values, diff, label=f"Difference: {custom_key.lower()}", lw=2)
    else:
        # Fallback if no legacy data exists
        for key, val in custom_data.items():
            ax.plot(t_values, val, label=key)

    ax.set_yscale("log")
    ax.set_ylabel(r"Absolute Difference |Custom - Legacy|")
    ax.set_xlabel(r"Interpolation Parameter $t$")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add parameter label box
    fig.text(0.85, 0.5, get_param_label(start_params, end_params), verticalalignment='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), transform=fig.transFigure)

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    save_plot_with_timestamp(fig)
    plt.show()