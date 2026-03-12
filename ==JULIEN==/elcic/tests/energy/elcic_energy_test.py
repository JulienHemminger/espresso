import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs


@pytest.fixture(scope="module")
def system():
    """Initializes the ESPResSo system singleton once for the session."""
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges=[+1, -1]):
    """Resets and reconfigures the existing ESPResSo system."""
    system.part.clear()
    system.electrostatics.clear()

    system.box_l = [box_l, box_l, box_l + gap_size]
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    half_box_l = box_l / 2.0
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z], q=charges[0])
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z + r_p1_p2], q=charges[1])
    return system


def calculate_analytic(z, dist, prefactor, delta_mid_bot):
    """Calculates analytic energy for q=1 at a specific z."""
    energy = prefactor * (
        -1 / dist
        + delta_mid_bot * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )
    return energy


def run(
    system,
    box_l,
    gap_size,
    prefactor,
    p1_pos_z,
    r_p1_p2,
    delta_mid_top,
    delta_mid_bot,
    charges=[+1, -1],
):
    """LEGACY ELC
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=acc, check_neutrality=False
    )
    elc = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=gap_size,
        maxPWerror=acc,
        delta_mid_bot=delta_mid_bot,
        delta_mid_top=delta_mid_top,
        neutralize=False,
    )
    system.electrostatics.solver = elc
    elc_total = system.analysis.energy()["total"]
    """
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8]
    errors = []

    # New regrouped storage
    comp_data = {"Total e_3d": [], "Total e_corr": [], "e_far": []}

    ana_energy = calculate_analytic(p1_pos_z, r_p1_p2, prefactor, delta_mid_bot)

    for acc in accuracies:
        setup_system(
            system,
            box_l,
            gap_size,
            p1_pos_z,
            r_p1_p2,
            charges,
        )

        # Get detailed contributions
        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )

        # Calculate totals using ELC inclusion-exclusion formula coefficients: 0.5 * (Lt - Pm1 + L0)
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
        "p1_pos_z": 0.1,
        "r_p1_p2": 2.0,
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
