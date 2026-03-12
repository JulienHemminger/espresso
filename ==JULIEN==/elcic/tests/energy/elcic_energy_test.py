import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs


@pytest.fixture(scope="module")
def system():
    """Initializes the ESPResSo system singleton."""
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system_random(system, box_l, gap_size, min_distance, charges=[+1, -1]):
    """Resets system and places two particles at a fixed distance in random positions."""
    system.part.clear()
    system.electrostatics.clear()

    # The actual Z-boundary including gap
    full_box_z = box_l + gap_size
    system.box_l = [box_l, box_l, full_box_z]
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    # 1. Generate random position for Particle 1 (p1)
    # We leave a buffer of 'min_distance' to ensure p2 doesn't go out of bounds
    p1_pos = np.random.uniform(low=min_distance, high=box_l - min_distance, size=3)

    # 2. Generate a random unit vector for direction
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)

    dx = min_distance * np.sin(theta) * np.cos(phi)
    dy = min_distance * np.sin(theta) * np.sin(phi)
    dz = min_distance * np.cos(theta)

    p2_pos = p1_pos + np.array([dx, dy, dz])

    # Add particles
    system.part.add(pos=p1_pos, q=charges[0])
    system.part.add(pos=p2_pos, q=charges[1])

    return system, p1_pos, p2_pos


def calculate_analytic(p1_pos, p2_pos, prefactor, delta_mid_bot):
    """Calculates analytic energy for q=1 based on particle positions."""
    # Vertical distance (dist) and height above bottom (z)
    dist = np.linalg.norm(p1_pos - p2_pos)
    z1 = p1_pos[2]
    z2 = p2_pos[2]

    # Using the existing formula logic relative to the lower particle (z_min)
    z_min = min(z1, z2)

    energy = prefactor * (
        -1 / dist
        + delta_mid_bot
        * (1 / (4 * z_min) - 1 / (2 * z_min + dist) + 1 / (4 * (z_min + dist)))
    )
    return energy


def run(
    system,
    box_l,
    gap_size,
    prefactor,
    min_distance,  # Replaced p1_pos_z and r_p1_p2
    delta_mid_top,
    delta_mid_bot,
    charges=[+1, -1],
):
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8]
    errors = []
    comp_data = {"Total e_3d": [], "Total e_corr": [], "e_far": []}

    # Setup system once to get the random positions for this 'run'
    system, p1_pos, p2_pos = setup_system_random(
        system, box_l, gap_size, min_distance, charges
    )

    # Calculate analytic energy based on the generated positions
    ana_energy = calculate_analytic(p1_pos, p2_pos, prefactor, delta_mid_bot)

    for acc in accuracies:
        # Re-apply setup for each accuracy (keeping positions consistent for this run)
        system.part.clear()
        system.part.add(pos=p1_pos, q=charges[0])
        system.part.add(pos=p2_pos, q=charges[1])

        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )

        e_3d_total = 0.5 * (
            contribs["lt"]["e_3d"] - contribs["pm1"]["e_3d"] + contribs["l0"]["e_3d"]
        )
        e_corr_total = 0.5 * (
            contribs["lt"]["e_corr"]
            - contribs["pm1"]["e_corr"]
            + contribs["l0"]["e_corr"]
        )
        e_far = contribs["e_far"]

        elc_total = e_3d_total + e_corr_total + e_far
        errors.append(abs(elc_total - ana_energy))

        comp_data["Total e_3d"].append(e_3d_total)
        comp_data["Total e_corr"].append(e_corr_total)
        comp_data["e_far"].append(e_far)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_labels = [f"{a:.0e}" for a in accuracies]
    x_pos = np.arange(len(x_labels))

    # Ax1: Stacked Bar Chart for regrouped contributions
    bottoms = np.zeros(len(accuracies))
    # Colors: Blue for 3D solver, Orange for parabolic correction, Green for Far-field images
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    for i, (label, vals) in enumerate(comp_data.items()):
        ax1.bar(
            x_pos,
            vals,
            bottom=bottoms,
            label=label,
            color=colors[i],
            alpha=0.7,
            width=0.5,
        )
        bottoms += np.array(vals)

    ax1.set_ylabel("Energy Contribution Value", fontsize=12)
    ax1.set_xlabel("P3M Target Accuracy", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Ax2: Error Scatter Plot (Right Y-axis)
    ax2 = ax1.twinx()
    ax2.scatter(
        x_pos,
        errors,
        color="black",
        marker="o",
        s=100,
        label="Total ELCIC Error",
        zorder=5,
    )
    ax2.set_ylabel("Absolute Error vs Analytic", color="black", fontsize=12)
    ax2.set_yscale("log")

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )

    plt.title(
        f"ELCIC Contribution Breakdown & Convergence\n$\Delta_{{bot}}={delta_mid_bot}, \Delta_{{top}}={delta_mid_top}$"
    )
    plt.tight_layout()
    plt.show()


def test_all(system):
    params = {
        "box_l": 4.0,
        "gap_size": 1.5,
        "prefactor": 2.0,
        "min_distance": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+1, -1],
    }

    # SINGLE PLATE
    # run(system, **params)  # neutral, metallic, PASS

    params["delta_mid_bot"] = 0.9
    # run(system, **params)  # neutral, non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    # run(system, **params)  # non-neutral, non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, metallic, FAIL - legacy elc doesnt work

    # DOUBLE PLATES
    params["charges"] = [+1, -1]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # neutral, both metallic, PASS

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    run(system, **params)  # neutral, both non-metallic, PASS

    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # neutral, mixed metallic + non-metallic, PASS

    """
    params["charges"] = [+1.2, -0.7]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # non-neutral, both metallic, FAIL - legacy elc doesnt work

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    # run(system, **params)  # non-neutral, both non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, mixed metallic + non-metallic, FAIL - legacy elc doesnt work
    """
