import numpy as np
import matplotlib.pyplot as plt
import espressomd
import copy
from datetime import datetime
import mpld3

from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy as get_elcic_energy_old
from common.legacy.energy import get_legacy_energy
from elcic.energy.single_plate.neutral.metallic.analytical import (
    get_ewald_elcic_2d as analytical_single_plate_2d_ewald_elcic_energy,
)
ERROR_CODE = 1.0


def run_single_test(system, params):
    """Calculates custom and legacy errors for a single parameter configuration."""
    system.part.clear()
    try:
        system.box_l = [params["lx"], params["ly"], params["lz"]]
    except Exception as e:
        print(f"Error: {e}")
        return ERROR_CODE, ERROR_CODE
    
    for pos, q in zip(params["positions"], params["charges"]):
        system.part.add(pos=pos, q=q)

    # --- Legacy Energy with Error Handling ---
    legacy_energy = get_legacy_energy(
            system,
            params
        )
    
    # --- Custom Energy with Error Handling ---
    try:
        custom_energy = get_elcic_energy_old(
            system=system,
            params=params,
        )
    except Exception as e:
        print(f"Warning: Custom energy calculation failed. Error: {e}")
        custom_energy = legacy_energy + ERROR_CODE


    return np.abs(custom_energy - legacy_energy), 0.0

# Helper to generate 7 points centered around a default
def get_sweep(default, step, min_val=None, max_val=None):
    value_count = 7
    step = 5.0 * step

    vals = np.linspace(default - value_count//2 * step, default + value_count//2 * step, value_count)
    if min_val is not None:
        vals = np.maximum(vals, min_val)
    if max_val is not None:
        vals = np.minimum(vals, max_val)
    return vals

def create_plot(system, params):
    plot_configs = [
        # Plot 1: Geometry (Steps of 0.5)
        {
            "lx": get_sweep(params["lx"], 0.5, min_val=10.0),
            "ly": get_sweep(params["ly"], 0.5, min_val=10.0),
            "lz": get_sweep(params["lz"], 0.5, min_val=17.5),
            "gap_size": get_sweep(params["gap_size"], 0.5, min_val=7.5, max_val=19),
        },
        # Plot 2: Physics (Prefactor steps 0.2, Delta steps 0.1)
        {
            "prefactor": get_sweep(params["prefactor"], 0.2, min_val=1.0, max_val=5.0),
            "delta_mid_top": get_sweep(params["delta_mid_top"], 0.1, min_val=-1.0, max_val=1.0),
            "delta_mid_bot": get_sweep(params["delta_mid_bot"], 0.1, min_val=-1.0, max_val=1.0),
        },
        # Plot 3: All Particle Coordinates (Steps of 0.1)
        {
            "p1_x": get_sweep(params["positions"][0][0], 0.1),
            "p1_y": get_sweep(params["positions"][0][1], 0.1),
            "p1_z": get_sweep(params["positions"][0][2], 0.1),
            "p2_x": get_sweep(params["positions"][1][0], 0.1),
            "p2_y": get_sweep(params["positions"][1][1], 0.1),
            "p2_z": get_sweep(params["positions"][1][2], 0.1),
        },
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.4)

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        for param_name, values in config.items():
            custom_errs = []
            legacy_errs = []

            for val in values:
                test_params = copy.deepcopy(params)

                # Mapping string keys to the position array
                pos_map = {
                    "p1_x": (0, 0),
                    "p1_y": (0, 1),
                    "p1_z": (0, 2),
                    "p2_x": (1, 0),
                    "p2_y": (1, 1),
                    "p2_z": (1, 2),
                }

                if param_name in pos_map:
                    p_idx, coord_idx = pos_map[param_name]
                    test_params["positions"][p_idx][coord_idx] = val
                else:
                    test_params[param_name] = val

                c_err, l_err = run_single_test(system, test_params)
                custom_errs.append(c_err)
                legacy_errs.append(l_err)

            # Plotting
            (line,) = ax.plot(
                values, custom_errs, marker="o", label=f"Custom ({param_name})"
            )
            ax.plot(
                values,
                legacy_errs,
                marker="x",
                linestyle="--",
                color=line.get_color(),
                alpha=0.6,
                label=f"Legacy ({param_name})",
            )

        ax.set_yscale("log")
        ax.set_title(f"Subplot {i + 1}: Sensitivity Analysis")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Absolute Error")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax.grid(True, which="both", ls="-", alpha=0.2)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"numgrad_{timestamp}.html"

        # 3. Save as interactive HTML
        plt.tight_layout()
        mpld3.save_html(fig, "/home/main/"+filename)


    plt.show()



# --- Configuration ---
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"


end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10



create_plot(system, end_params)