import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

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


def save_plot_with_timestamp(fig, base_directory="/home/main/"):
    """
    Saves the provided figure as a PNG with a timestamped filename.
    Ensures the directory exists before saving.
    """
    # Create the timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lerp2d_{timestamp}.png"
    
    # Ensure the directory exists (optional, but good practice)
    if not os.path.exists(base_directory):
        print(f"Directory {base_directory} not found. Saving to current directory.")
        full_path = filename
    else:
        full_path = os.path.join(base_directory, filename)
    
    # Save the figure
    # bbox_inches='tight' is recommended to prevent clipping of labels/legends
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"Figure successfully saved to: {full_path}")

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
        if get_legacy_energy:
            legacy_res = get_legacy_energy(system, params)
            results["legacy"].append(legacy_res)
        custom_res = get_custom_energy(system, params)
        results["custom"].append(custom_res)

    # --- Print Summary for t=1 (last result) ---
    if get_legacy_energy and results["legacy"]:
        for i in range(steps):
            t = t_values[i]
            params = lerp_dict(start_params, end_params, t)
            print(f"Parameters={params}")
            print(f"{'Name':<12} | {'Legacy':<10} | {'Custom':<10} | {'Error (abs)'}")
            print("-" * 50)
            leg_final = results["legacy"][i]
            cus_final = results["custom"][i]
            
            for key in cus_final.keys():
                if key in leg_final:
                    val_l = leg_final[key]
                    val_c = cus_final[key]
                    error = abs(val_l - val_c)
                    print(f"{key:<12} | {val_l:<15.10f} | {val_c:<15.10f} | {error:<15.10f}")

    # --- Prepare Data ---
    # Convert list of dicts to a dict of numpy arrays
    custom_data = {k: np.array([d[k] for d in results["custom"]]) for k in results["custom"][0].keys()}
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if get_legacy_energy and len(results["legacy"]) > 0:
        legacy_data = {k: np.array([d[k] for d in results["legacy"]]) for k in results["legacy"][0].keys()}
        
        # Identify common keys
        common_keys = [k for k in custom_data.keys() if k in legacy_data.keys()]
        
        for key in common_keys:
            # Calculate absolute difference for the plot
            diff = np.abs(custom_data[key] - legacy_data[key])
            ax.plot(t_values, diff, label=f"Difference: {key}", lw=2)
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