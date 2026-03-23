import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from src.energy.get_elcic_energy import get_elcic_energy_contribs
from src.energy.third_party.get_legacy_elc import get_legacy_elc_energy


@pytest.fixture(scope="module")
def system():
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, lx, ly, lz, gap_size, positions, charges):
    system.part.clear()
    system.box_l = [lx, ly, lz + gap_size]
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
    system.time_step = 0.01

    for i in range(len(charges)):
        system.part.add(pos=positions[i], q=charges[i])
    return system


def run(
    system,
    lx,
    ly,
    lz,
    gap_size,
    prefactor,
    delta_mid_top,
    delta_mid_bot,
    positions,
    charges,
    params,
):
    setup_system(system, lx, ly, lz, gap_size, positions, charges)

    accuracies = [
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11,
    ]  # crashes starting at 1e-12
    elc_errors = []
    contrib_data = {"e_3d": [], "e_corr": [], "e_far": []}

    ana_energy = (
        -0.2704042280997297
    )  # n_max=2^10: ana_energy=-0.2704042280997297, in 8:30min

    for acc in accuracies:
        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )
        contrib_data["e_3d"].append(
            sum(contribs[k]["e_3d"] for k in ["l0", "pm1", "lt"])
        )
        contrib_data["e_corr"].append(
            sum(contribs[k]["e_corr"] for k in ["l0", "pm1", "lt"])
        )
        contrib_data["e_far"].append(contribs["e_far"])

        energy_legacy = get_legacy_elc_energy(
            system, gap_size, acc, delta_mid_top, delta_mid_bot
        )

        elc_errors.append(np.abs(energy_legacy - ana_energy))

    # PLOTTING
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
    bottoms = np.zeros(len(accuracies))
    bar_width = 0.2 * np.array(accuracies)

    # 1. Secondary Axis: Energy Contributions (Stacked Bars)
    for i, (label, vals) in enumerate(contrib_data.items()):
        ax2.bar(
            accuracies,
            vals,
            bottom=bottoms,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.3,
            edgecolor="grey",
        )
        bottoms += np.array(vals)

    # 2. Primary Axis: Errors (Lines)
    ax1.loglog(
        accuracies,
        elc_errors,
        "o-",
        label="ELC Error",
        color="blue",
        linewidth=2,
        zorder=5,
    )

    ax1.loglog(accuracies, accuracies, "k:", alpha=0.5, label="Target Accuracy (1:1)")

    # Formatting
    ax1.set_xlabel("Requested Accuracy (pw_error)")
    ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
    ax2.set_ylabel("Component Value (Linear Scale)", color="#7f8c8d")
    plt.title("Accurcay Convergence")

    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + bars,
        labels + bar_labels,
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
    )

    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.invert_xaxis()
    fig.tight_layout()
    plt.show()


def test_all(system):
    params = {
        "lx": 9.0,
        "ly": 12.0,
        "lz": 19.0,
        "gap_size": 15.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+1, -1],
    }
    params["positions"] = [np.array([7, 1, 3]), np.array([4, 5, 2])]

    run(system, **params, params=params)
