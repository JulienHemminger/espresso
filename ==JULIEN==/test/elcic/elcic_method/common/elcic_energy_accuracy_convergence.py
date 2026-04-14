import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs
from elc.energy.legacy_elc_energy import get_legacy_elc_energy


def plot_convergence(accuracies, error_data, contrib_data, params):
    """Handles the visualization of error convergence and energy components."""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
    bottoms = np.zeros(len(accuracies))
    # Adjust bar width relative to log scale points
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

    # Plotting the Delta Error (Custom vs Legacy)
    ax1.loglog(
        accuracies,
        error_data,
        "o-",
        label="|Custom - Legacy|",
        color="red",
        lw=2,
        zorder=5,
    )
    
    ax1.loglog(accuracies, accuracies, "k:", alpha=0.5, label="Target Accuracy (1:1)")

    # Format params dict for display
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items() if not isinstance(v, list)])

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
    ax1.set_ylabel("Error vs Legacy (Log Scale)", color="red")
    ax2.set_ylabel("Component Value (Linear Scale)", color="#7f8c8d")
    plt.title("Convergence of Custom ELCIC against Legacy ELC")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.05, 1))

    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.invert_xaxis()
    fig.tight_layout()
    plt.show()


def run(system, params):
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    for pos, q in zip(params["positions"], params["charges"]):
        system.part.add(pos=pos, q=q)

    accuracies = [10**-i for i in range(1, 7)] 
    custom_vs_legacy_errors = []
    bar_plot_data = {"e_3d": [], "e_corr": [], "e_far": []}

    for acc in accuracies:
        contribs = get_elcic_energy_contribs(
            system,
            params["gap_size"],
            acc,
            params["prefactor"],
            params["delta_mid_bot"],
            params["delta_mid_top"],
        )

        # Aggregate contributions for the secondary axis bars
        keys = ["l0", "pm1", "lt"]
        bar_plot_data["e_3d"].append(sum(contribs[k]["e_3d"] for k in keys))
        bar_plot_data["e_corr"].append(sum(contribs[k]["e_corr"] for k in keys))
        bar_plot_data["e_far"].append(contribs["e_far"])

        # Calculate Energies
        custom_energy = contribs["e_far"] + contribs["e_near"]
        legacy_energy = get_legacy_elc_energy(
            system, 
            params["gap_size"], 
            params["prefactor"],
            acc, 
            params["delta_mid_top"], 
            params["delta_mid_bot"]
        )

        # Legacy as ground truth
        custom_vs_legacy_errors.append(custom_energy - legacy_energy)

    plot_convergence(accuracies, custom_vs_legacy_errors, bar_plot_data, params)