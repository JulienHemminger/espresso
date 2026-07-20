import matplotlib.pyplot as plt
import numpy as np
from src.common.generators.position_generator import get_rdm_constrained_points_np
from src.common.plot_saving import save_plot_with_timestamp
from src.elc.force.analytical_elc_forces import get_ewald_forces_2d
from src.elc.force.custom_elc_forces import get_elc_forces_contribs


def run_accuracy_convergence(
    system,
    prefactor=1.0,
    gap_size=1.0,
    charges=[+1.0, -1.0],
    show_convergence_plot=True,
):
    # Setup parameters based on system type

    # accuracies = np.logspace(-4, -5, num=2)
    accuracies = np.logspace(-4, -8, num=5)
    # accuracies = np.logspace(-4, -12, num=9)

    lx, ly, lz = system.box_l
    particle_count = len(charges)
    positions = get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count
    )

    elc_errors = []
    contrib_data = {
        r"$\mathrm{F_{3D}}$": [],
        r"$\mathrm{F_{dipole}}$": [],
        r"$\mathrm{F_{far}}$": [],
    }

    for acc in accuracies:
        system.part.clear()
        for i in range(particle_count):
            system.part.add(pos=positions[i], q=charges[i])

        ana_forces = get_ewald_forces_2d(system, n_max=100, prefactor=prefactor)

        pref, f_3d, f_elc_recip, f_corr_moments = get_elc_forces_contribs(
            system, gap_size, acc, prefactor
        )

        f_final = list(f_3d + pref * (f_elc_recip + f_corr_moments))

        i = 0  # choose a random particle
        elc_errors.append(
            abs(
                np.linalg.norm(f_final[i] - ana_forces[i])
                / np.linalg.norm(ana_forces[i])
            )
        )  # Vector L2 Relative Error

        contrib_data[r"$\mathrm{F_{3D}}$"].append(np.linalg.norm(f_3d[i]))
        contrib_data[r"$\mathrm{F_{dipole}}$"].append(
            pref * np.linalg.norm(f_corr_moments[i])
        )
        contrib_data[r"$\mathrm{F_{far}}$"].append(
            pref * np.linalg.norm(f_elc_recip[i])
        )

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()

        colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
        bottoms = np.zeros(len(accuracies))
        bar_width = 1.0 * np.array(accuracies)

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
            label=r"$\mathrm{|Ewald\ 2D - Custom\ ELC|}$",
            color="#2980b9",
            linewidth=2,
            zorder=5,
        )

        # Formatting
        ax1.set_xlabel("Requested Accuracy")
        ax1.set_ylabel("Error", color="#2980b9")
        ax2.set_ylabel("Force Contribution Magnitude", color="black")

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

        save_plot_with_timestamp(fig)
        plt.show()


import espressomd
import espressomd.electrostatics

system = espressomd.System(box_l=[10, 10, 10])
system.time_step = 0.01
system.cell_system.skin = 0.4


run_accuracy_convergence(
    system, prefactor=1, gap_size=1, charges=[+1, -1], show_convergence_plot=True
)
