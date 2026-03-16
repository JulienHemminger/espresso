import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest


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


def calculate_analytic_alex(z, dist, prefactor, delta_mid_bot):
    """Calculates analytic energy for q=1 at a specific z."""
    return prefactor * (  # elc_vs_analytic.py
        -1 / dist
        + delta_mid_bot * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )


def calculate_elcic_energy(params):
    """
    Computes the total electrostatic energy for a 2D+h system
    with two dielectric interfaces based on Tyagi et al. (2008).
    """
    box_l = params["box_l"]
    prefactor = params["prefactor"]
    dt = params["delta_mid_top"]
    db = params["delta_mid_bot"]
    delta = dt * db

    # Particle properties
    q = params["charges"]
    z = [params["p1_pos_z"], params["p1_pos_z"] + params["r_p1_p2"]]

    energy = 0.0
    # Interaction between the two real charges
    r_12 = abs(z[0] - z[1])
    energy += prefactor * q[0] * q[1] / r_12

    # Interaction with image charges (direct summation)
    # We sum over 'n' generations of reflections.
    # For delta < 1, this converges quickly.
    max_gen = 100

    for i in range(2):
        for j in range(2):
            qi, qj = q[i], q[j]
            zi, zj = z[i], z[j]

            # Sum over infinite image sequences defined in the paper
            # Lower dielectric sequences (Eq. 2.3 & 2.4)
            for n in range(max_gen):
                # Image at -(2*n*box_l + zj) with charge qj * (delta^n * db)
                pos_down1 = -(2 * n * box_l + zj)
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * db)) / abs(zi - pos_down1)
                )

                # Image at -(2*(n+1)*box_l - zj) with charge qj * delta^(n+1)
                if n < max_gen - 1:  # Avoid double counting or out of range
                    pos_down2 = -(2 * (n + 1) * box_l - zj)
                    energy += (
                        0.5
                        * prefactor
                        * qi
                        * (qj * delta ** (n + 1))
                        / abs(zi - pos_down2)
                    )

            # Upper dielectric sequences (Eq. 2.5 & 2.6)
            for n in range(max_gen):
                # Image at (2*(n+1)*box_l - zj) with charge qj * (delta^n * dt)
                pos_up1 = 2 * (n + 1) * box_l - zj
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * dt)) / abs(zi - pos_up1)
                )

                # Image at (2*(n+1)*box_l + zj) with charge qj * delta^(n+1)
                pos_up2 = 2 * (n + 1) * box_l + zj
                energy += (
                    0.5 * prefactor * qi * (qj * delta ** (n + 1)) / abs(zi - pos_up2)
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
    params={},
):
    """Executes the simulation for multiple accuracies and plots results."""
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    errors = []

    # Storage for stacked bar chart

    # Set up the base system geometry
    setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges)
    # ana_energy = calculate_analytic_alex(p1_pos_z, r_p1_p2, prefactor, delta_mid_bot)
    ana_energy = calculate_elcic_energy(params)

    for acc in accuracies:
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
        errors.append(abs(elc_total - ana_energy))

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 7))  # Slightly wider for long labels
    x_labels = [f"{a:.0e}" for a in accuracies]
    x_pos = np.arange(len(x_labels))

    ax1.set_ylabel("Energy Contribution Value", fontsize=12)
    # ... (rest of the formatting code remains the same)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc="upper left")

    # 2. Scatter Plot (Accuracy Error) on Right Axis
    ax2 = ax1.twinx()
    ax2.scatter(
        x_pos, errors, color="black", marker="D", s=100, label="Abs Error", zorder=5
    )
    ax2.set_ylabel("Error |ELC - Analytic|", color="red")
    ax2.set_yscale("log")

    title = f"ELCIC Convergence (Bot={delta_mid_bot}, Top={delta_mid_top})"
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


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
