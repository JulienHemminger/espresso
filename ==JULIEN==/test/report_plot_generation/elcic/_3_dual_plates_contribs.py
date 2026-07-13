import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy

# Configuration
params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
}
params["lz"] = params["gap_size"] + 40


def run_z_variation_plot(system, params, z_start=3.0, z_end=13.0, steps=16):
    z_values = np.linspace(z_start, z_end, steps)

    results = {"legacy": [], "e_total": [], "e_near": [], "e_far": []}

    # Run Simulation/Evaluation Loop
    for z in z_values:
        system.electrostatics.clear()
        system.part.clear()
        system.box_l = [params["lx"], params["ly"], params["lz"]]

        # Update the z-position of the first particle dynamically
        current_positions = [p.copy() for p in params["positions"]]
        current_positions[0][2] = z

        for i in range(len(params["charges"])):
            system.part.add(pos=current_positions[i], q=params["charges"][i])

        # Prepare evaluation parameters dict
        eval_params = params.copy()
        eval_params["positions"] = current_positions

        print("==============================================")
        print(f"Evaluating at Particle 0 z = {z:.4f}")

        # Compute Legacy Energy
        legacy_energy = get_legacy_energy(system, eval_params)
        results["legacy"].append(legacy_energy)

        # Compute Custom Energy
        custom_res = get_elcic_energy(system, eval_params)

        results["e_total"].append(custom_res["e_total"])
        results["e_near"].append(custom_res["e_near"])
        results["e_far"].append(custom_res["e_far"])

        print(
            f"custom_implementation_energy = {custom_res['e_total']}, error={abs(custom_res['e_total'] - legacy_energy)}"
        )
        print(f"ground_truth_energy = {legacy_energy}")

    # Convert results to arrays
    for key in results:
        results[key] = np.array(results[key])

    e_total = np.array(results["e_total"])
    e_legacy = np.array(results["legacy"])

    absolute_error = np.abs(e_total - e_legacy)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Main Energies
    (l1,) = ax1.plot(
        z_values,
        results["e_total"],
        label="Custom Total",
        color="blue",
        lw=2,
        marker="x",
    )
    (l2,) = ax1.plot(z_values, results["e_near"], label="e_near", color="cyan", ls="--")
    (l3,) = ax1.plot(
        z_values,
        results["legacy"],
        label="Legacy Ground Truth",
        color="red",
        lw=2,
        marker="+",
    )

    ax1.set_xlabel(r"Particle $z$ Position")
    ax1.set_ylabel("Energy (Main)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # First Right Axis: e_far
    ax_far = ax1.twinx()
    (l4,) = ax_far.plot(
        z_values, results["e_far"], label="e_far", color="teal", ls="--"
    )
    ax_far.set_ylabel("Energy (e_far)", color="teal")
    ax_far.tick_params(axis="y", labelcolor="teal")

    # Second Right Axis: Absolute Error (spinned out to the right)
    ax_err = ax1.twinx()
    ax_err.spines["right"].set_position(("outward", 60))
    (l5,) = ax_err.plot(
        z_values,
        absolute_error,
        label="Absolute Error",
        color="purple",
        ls=":",
        marker="o",
    )
    ax_err.set_yscale("log")
    ax_err.set_ylabel("Absolute Error (|Custom - Legacy|)", color="purple")
    ax_err.tick_params(axis="y", labelcolor="purple")

    # Consolidated Legend
    lines = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize="small")

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Initialize System
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Run the routine tracking the actual physical z coordinates
run_z_variation_plot(
    system=system,
    params=params,
    z_start=3.0,
    z_end=13.0,
    steps=4,
)


# show parts of E_near = 0.5 * (e_lt - e_pm1 + e_l0)
