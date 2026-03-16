import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from elcic.src.energy.get_analytical_energy import calculate_elcic_energy
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs

# Assuming the file containing get_elcic_energy_contribs is elcic_utils.py


@pytest.fixture(scope="module")
def system():
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges=[+1, -1]):
    system.part.clear()
    system.box_l = [box_l, box_l, box_l + gap_size]
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
    system.time_step = 0.01

    half_box_l = box_l / 2.0
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z], q=charges[0])
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z + r_p1_p2], q=charges[1])
    return system


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
    params={},
):
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    errors_legacy = []
    errors_elcic = []

    # Storage for stacked bars
    e_3d_sums = []
    e_corr_sums = []
    e_far_vals = []

    setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges)
    ana_energy = calculate_elcic_energy(params)

    for acc in accuracies:
        # 1. Legacy ELC Calculation
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
        legacy_energy = system.analysis.energy()["total"]
        errors_legacy.append(abs(legacy_energy - ana_energy))

        # 2. ELCIC Decomposition Calculation
        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )

        # Aggregate components for the bar chart
        e_3d_sums.append(
            contribs["l0"]["e_3d"] + contribs["pm1"]["e_3d"] + contribs["lt"]["e_3d"]
        )
        e_corr_sums.append(
            contribs["l0"]["e_corr"]
            + contribs["pm1"]["e_corr"]
            + contribs["lt"]["e_corr"]
        )
        e_far_vals.append(contribs["e_far"])

        # Calculate ELCIC specific error
        elcic_total = contribs["e_near"] + contribs["e_far"]
        errors_elcic.append(abs(elcic_total - ana_energy))

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(accuracies))

    # Stacked Bar Chart (behind the points)
    ax1.bar(x_pos, e_3d_sums, label="Sum E_3D", alpha=0.3, color="blue")
    ax1.bar(
        x_pos,
        e_corr_sums,
        bottom=e_3d_sums,
        label="Sum E_Corr",
        alpha=0.3,
        color="green",
    )
    ax1.bar(
        x_pos,
        e_far_vals,
        bottom=np.array(e_3d_sums) + np.array(e_corr_sums),
        label="E_Far",
        alpha=0.3,
        color="orange",
    )

    ax1.set_ylabel("Energy Components (Sum of Sets)", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{a:.0e}" for a in accuracies])
    ax1.legend(loc="upper left")

    # Dual Scatter Plots for Error
    ax2 = ax1.twinx()
    ax2.scatter(
        x_pos,
        errors_legacy,
        color="black",
        marker="D",
        s=80,
        label="Legacy Error",
        zorder=5,
    )
    ax2.scatter(
        x_pos,
        errors_elcic,
        color="red",
        marker="o",
        s=80,
        label="ELCIC Error",
        zorder=5,
    )

    ax2.set_ylabel("Absolute Error vs Analytic", color="black")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")

    plt.title(
        f"ELCIC Convergence & Energy Breakdown (Bot={delta_mid_bot:.2f}, Top={delta_mid_top:.2f})"
    )
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    # --- Print Bar Values ---
    print("\n" + "=" * 50)
    print(f"{'Accuracy':<10} | {'Sum E_3D':<12} | {'Sum E_Corr':<12} | {'E_Far':<12}")
    print("-" * 50)
    for i, acc in enumerate(accuracies):
        print(
            f"{acc:<10.0e} | {e_3d_sums[i]:<12.6f} | {e_corr_sums[i]:<12.6f} | {e_far_vals[i]:<12.6f}"
        )
    print("=" * 50 + "\n")


def test_all(system):
    params = {
        "box_l": 200.0,
        "gap_size": 75.0,
        "prefactor": 2.0,
        "p1_pos_z": 10.0,
        "r_p1_p2": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": 39.0 / 41.0,
        "charges": [+1, -1],
    }
    run(system, **params, params=params)

    """
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
    # run(system, **params)  # neutral, both non-metallic, PASS

    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # neutral, mixed metallic + non-metallic, PASS

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
