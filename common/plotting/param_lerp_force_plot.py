import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Configuration for easier maintenance
NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor", "lambda"]
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
    full_path = os.path.join(base_directory, filename)
    
    # Save the figure
    # bbox_inches='tight' is recommended to prevent clipping of labels/legends
    fig.savefig(full_path, bbox_inches='tight', dpi=300)

def run_lerp_plot(system, start_params, end_params, get_custom_force, get_legacy_force, get_analytical_force=None, steps=20): 
    t_values = np.linspace(0, 1, steps)
    # Store force vectors: list of (2, 3) arrays
    results = {"legacy": [], "custom": []}

    for t in t_values:
        system.electrostatics.clear()
        params = lerp_dict(start_params, end_params, t)
        
        system.part.clear()
        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])
        
        # Assume these return np.array of shape (2, 3)
        res_leg = get_legacy_force(system, params) 
        res_custom = get_custom_force(system, params) 
        
        results["legacy"].append(res_leg)
        results["custom"].append(res_custom)

    # Convert to (steps, 2, 3) arrays
    data_leg = np.array(results["legacy"])
    data_cust = np.array(results["custom"])

    # --- Plotting ---
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 1. Top Subplot: Force Magnitudes (L2 Norm of each particle)
    
    # Define a explicit visual mapping to maximize contrast
    # (Color, Linestyle, Marker, Marker_Start_Offset)
    style_mapping = {
        # Particle 0: Blues / Cool tones
        "legacy_0": {"color": "#1f77b4", "ls": "--", "marker": "o", "offset": 0}, # Thick dashed blue with circles
        "custom_0": {"color": "#17becf", "ls": "-",  "marker": "x", "offset": 2}, # Thin solid cyan with crosses
        
        # Particle 1: Oranges / Warm tones
        "legacy_1": {"color": "#d62728", "ls": ":",  "marker": "s", "offset": 1}, # Thick dotted red with squares
        "custom_1": {"color": "#ff7f0e", "ls": "-.", "marker": "^", "offset": 3}, # Thin dash-dot orange with triangles
    }

    for i in range(2):
        mag_leg = np.linalg.norm(data_leg[:, i, :], axis=1)
        mag_cust = np.linalg.norm(data_cust[:, i, :], axis=1)
        
        leg_style = style_mapping[f"legacy_{i}"]
        cust_style = style_mapping[f"custom_{i}"]
        
        # Legacy Line: Thick backdrop line
        ax1.plot(t_values, mag_leg, 
                 label=f"Legacy P{i}", 
                 color=leg_style["color"],
                 ls=leg_style["ls"], 
                 lw=4,                 
                 alpha=0.6,            
                 marker=leg_style["marker"], 
                 markersize=6, 
                 markevery=(leg_style["offset"], 4))     
        
        # Custom Line: Sits cleanly inside the legacy line
        ax1.plot(t_values, mag_cust, 
                 label=f"Custom P{i}", 
                 color=cust_style["color"],
                 ls=cust_style["ls"], 
                 lw=2,                 
                 alpha=0.9, 
                 marker=cust_style["marker"], 
                 markersize=6, 
                 markevery=(cust_style["offset"], 4))
    
    ax1.set_ylabel("Force Magnitude |F|")
    ax1.legend(fontsize='small', loc='upper right')
    ax1.set_title("Comparison of Force Magnitudes")
    ax1.grid(True, which="both", alpha=0.3)

    # 2. Bottom Subplot: Error (Norm of the difference vector per particle)
    diff = data_cust - data_leg
    for i in range(2):
        err = np.linalg.norm(diff[:, i, :], axis=1)
        print(f"For particle {i}, Max. Force Error "+"|F_{custom} - F_{truth}| = "+f"{np.max(err)} - is only satisfactory if below 1e-8")
        # Added distinct markers here too in case error profiles match exactly
        ax3.plot(t_values, err, 
                 label=f"Err P{i}", 
                 lw=2, 
                 marker="d" if i == 0 else "s", 
                 markersize=5, 
                 markevery=2)
    
    ax3.set_yscale("log")
    ax3.set_ylabel("Force Error ($|F_{cust} - F_{leg}|$)")
    ax3.set_xlabel(r"Interpolation Parameter $t$")
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.3)

    # Add text box configuration
    param_text = get_param_label(start_params, end_params)
    fig.text(0.84, 0.5, param_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    save_plot_with_timestamp(fig) 
    plt.show()