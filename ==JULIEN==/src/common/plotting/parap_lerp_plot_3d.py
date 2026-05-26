import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import copy

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

def run_2d_contour_plot(system, base_params,
                        x_param_key, x_values,
                        y_param_key, y_values,
                        get_custom_energy,
                        get_analytical_energy=None,
                        get_legacy_energy=None):
   
    # Create meshgrid of the two parameters
    X, Y = np.meshgrid(x_values, y_values)
    error_matrix = np.zeros_like(X)

    # Evaluate energy for each grid point
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            # Build parameter dict for this grid point
            params = copy.deepcopy(base_params)
            params[x_param_key] = X[i, j]
            params[y_param_key] = Y[i, j]

            # Set up system (positions, charges, box)
            system.electrostatics.clear()
            system.part.clear()
            system.box_l = [params["lx"], params["ly"], params["lz"]]
            for k in range(len(params["charges"])):
                system.part.add(pos=params["positions"][k], q=params["charges"][k])

            ref_energy = 0
            if get_analytical_energy is not None:
                ref_energy = get_analytical_energy(params)
            elif get_legacy_energy is not None:
                ref_energy = get_legacy_energy(system, params)
            # Obtain energies
            custom_energy = get_custom_energy(system, params)["e_total"]


            
            error_matrix[i, j] = abs(custom_energy - ref_energy)
            

    # ----- Plotting -----
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use filled contours (contourf) with a nice colormap
    cmap = "viridis" if get_analytical_energy or get_legacy_energy else "RdBu_r"
    contour = ax.contourf(X, Y, error_matrix, levels=20, cmap=cmap)
    cbar = fig.colorbar(contour, ax=ax)
    if get_analytical_energy or get_legacy_energy:
        cbar.set_label("Absolute Error")
    else:
        cbar.set_label("Total Custom Energy")

    ax.set_xlabel(x_param_key)
    ax.set_ylabel(y_param_key)
    ax.set_title("2D parameter scan")

    # Add a text box with the fixed parameters (optional, like original)
    fixed_lines = ["**Fixed Parameters**"]
    for key, val in base_params.items():
        if key not in (x_param_key, y_param_key):
            if key == "positions":
                fixed_lines.append("positions:")
                for i, pos in enumerate(val):
                    fixed_lines.append(f"  P{i}: {np.array2string(np.array(pos), precision=1)}")
            else:
                fixed_lines.append(f"{key}: {val}")
    fig.text(0.85, 0.5, "\n".join(fixed_lines),
             verticalalignment='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             transform=fig.transFigure)

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    # Reuse your existing save function
    save_plot_with_timestamp(fig)
    plt.show()
