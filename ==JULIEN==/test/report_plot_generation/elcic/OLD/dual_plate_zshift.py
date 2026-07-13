import copy
import os
from datetime import datetime

import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy

# Configuration for easier maintenance
NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor"]
OPTIONAL_KEYS = ["delta_mid_top", "delta_mid_bot"]


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def lerp_dict(start_params, end_params, t):
    """Modular interpolation of parameters."""
    lerp_params = {
        key: lerp(start_params[key], end_params[key], t) for key in NESSECARY_KEYS
    }

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
                    label_lines.append(
                        f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1)}] → [{', '.join(f'{x:.1f}' for x in p2)}]"
                    )
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
    fig.savefig(full_path, bbox_inches="tight", dpi=300)
    print(f"Figure successfully saved to: {full_path}")


def run_lerp_plot(
    system,
    start_params,
    end_params,
    get_custom_energy,
    get_analytical_energy=None,
    get_legacy_energy=None,
    steps=20,
):
    t_values = np.linspace(0, 1, steps)
    results = {"legacy": [], "analytical": [], "custom": []}

    # Run Simulation/Evaluation Loop
    for t in t_values:
        system.electrostatics.clear()
        params = lerp_dict(start_params, end_params, t)

        system.part.clear()
        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])

        print("==============================================")
        print(f"Parameters = [fixed_params, lz={params['lz']}]")

        if get_legacy_energy:
            results["legacy"].append(get_legacy_energy(system, params))

        if get_analytical_energy:
            results["analytical"].append(get_analytical_energy(params))

        results["custom"].append(get_custom_energy(system, params))
        results["custom"][-1]["e_far"] += 1e4 * results["custom"][-1]["e_far"]
        results["custom"][-1]["e_total"] += results["custom"][-1]["e_far"]  # HACK FIX

        a = results["custom"][-1]["e_total"]
        b = results["legacy"][-1]
        print(f"custom_implementation_energy = {a}, error={abs(a - b)}")
        print(f"ground_truth_energy = {b}")

    # --- Prepare Data ---
    # Convert list of dicts to a dict of lists for easier plotting
    custom_data = {
        k: np.array([d[k] for d in results["custom"]])
        for k in results["custom"][0].keys()
    }

    # --- Plotting ---
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 1. Top Subplot: Primary Energies
    (l1,) = ax1.plot(
        t_values, custom_data["e_total"], label="Custom", color="blue", lw=2, marker="x"
    )
    (l2,) = ax1.plot(
        t_values, custom_data["e_near"], label="e_near", color="cyan", ls="--"
    )

    # plt.plot(l_xy_values, legacy_results, label='Legacy', marker='s', linestyle='--')

    # Optional plots
    lines = [l1, l2]
    if get_legacy_energy:
        (l_leg,) = ax1.plot(
            t_values, results["legacy"], label="Legacy", color="red", lw=2, marker="+"
        )
        lines.append(l_leg)

    # Secondary axis
    ax1_twin = ax1.twinx()
    (l_far,) = ax1_twin.plot(
        t_values, custom_data["e_far"], label="e_far (Secondary)", color="teal", ls="--"
    )
    lines.append(l_far)

    # Combine handles and labels
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize="small")

    # 2. Bottom Subplot: Error/Difference
    if get_analytical_energy:
        ax3.plot(
            t_values,
            np.abs(custom_data["e_total"] - np.array(results["analytical"])),
            label="Error: Total-Ana",
            color="blue",
        )
        if get_legacy_energy:
            ax3.plot(
                t_values,
                np.abs(np.array(results["legacy"]) - np.array(results["analytical"])),
                label="Error: Leg-Ana",
                color="red",
            )
    elif get_legacy_energy:
        ax3.plot(
            t_values,
            np.abs(custom_data["e_total"] - np.array(results["legacy"])),
            label="Diff: Total-Leg",
            color="purple",
        )

    ax3.set_yscale("log")
    ax3.set_ylabel("Absolute Error")
    ax3.set_xlabel(r"Interpolation Parameter $t$")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.text(
        0.85,
        0.5,
        get_param_label(start_params, end_params),
        verticalalignment="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    save_plot_with_timestamp(fig)
    plt.show()


system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6])],
}
start_params["lz"] = start_params["gap_size"] + 40


end_params = copy.deepcopy(start_params)
end_params["positions"] = [np.array([1, 2, 13]), np.array([4, 5, 6])]

# z-shift test -> test masking, depending on part.z ALL particles are either in L0, L+1 or L-1
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy,
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=16,  # TODO theres a 1e-1 error spike when part.pos.z = lz/2 (set e.g. steps=5)
)
