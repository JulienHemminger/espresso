import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

# Assuming these modules are in your python path
from src.energy.get_elcic_energy import get_elcic_energy_contribs
from src.energy.third_party.analytical_elcic_energy import analytical_elcic_energy
from src.energy.third_party.get_legacy_elc import get_legacy_elc_energy


def plot_convergence(accuracies, elc_errors, contrib_data, params):
    """Handles the visualization of error convergence and energy components."""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
    bottoms = np.zeros(len(accuracies))
    bar_width = 0.2 * np.array(accuracies)

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

    ax1.loglog(
        accuracies, elc_errors, "o-", label="ELC Error", color="blue", lw=2, zorder=5
    )
    ax1.loglog(accuracies, accuracies, "k:", alpha=0.5, label="Target Accuracy (1:1)")

    # Format params dict for display
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])

    # Add text box to bottom right
    ax1.text(
        0.95,
        0.05,
        param_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="silver"),
    )

    ax1.set_xlabel("Requested Accuracy (pw_error)")
    ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
    ax2.set_ylabel("Component Value (Linear Scale)", color="#7f8c8d")
    plt.title("Accuracy Convergence")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.05, 1))

    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.invert_xaxis()
    fig.tight_layout()
    plt.show()


def run_analysis():
    # Configuration
    params = {
        "lx": 50.0,
        "ly": 50.0,
        "lz": 50.0,
        "gap_size": 10.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+1, -1],
        "positions": [np.array([0, 0, 0.01]), np.array([1, 1, 0.02])],
    }

    system = espressomd.System(box_l=[params["lx"], params["ly"], params["lz"]])
    system.time_step = 0.01
    for pos, q in zip(params["positions"], params["charges"]):
        system.part.add(pos=pos, q=q)

    accuracies = [10**-i for i in range(4, 12)]
    elc_errors = []
    contrib_data = {"e_3d": [], "e_corr": [], "e_far": []}

    ana_energy = analytical_elcic_energy(system, params, tol=1e-12)
    print(f"Analytical Energy: {ana_energy:.10f}")

    for acc in accuracies:
        contribs = get_elcic_energy_contribs(
            system,
            params["gap_size"],
            acc,
            params["prefactor"],
            params["delta_mid_bot"],
            params["delta_mid_top"],
        )

        # Aggregate contributions
        keys = ["l0", "pm1", "lt"]
        contrib_data["e_3d"].append(sum(contribs[k]["e_3d"] for k in keys))
        contrib_data["e_corr"].append(sum(contribs[k]["e_corr"] for k in keys))
        contrib_data["e_far"].append(contribs["e_far"])

        energy_legacy = get_legacy_elc_energy(
            system,
            params["gap_size"],
            acc,
            params["delta_mid_top"],
            params["delta_mid_bot"],
        )
        elc_errors.append(np.abs(energy_legacy - ana_energy))

    plot_convergence(accuracies, elc_errors, contrib_data, params)


if __name__ == "__main__":
    run_analysis()
