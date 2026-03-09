import matplotlib.pyplot as plt
import numpy as np
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.common.has_downward_trend import has_downward_trend
from elc.src.forces.get_elc_forces import get_elc_forces_contribs
from elc.src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d


def run_accuracy_convergence(
    system,
    prefactor=1.0,
    gap_size=1.0,
    charges=[+1.0, -1.0],
    show_convergence_plot=True,
    *args,
    **kwargs,
):
    # Setup parameters based on system type
    pw_errors = np.logspace(-4, -8, num=5)
    title = "Force Accuracy Convergence"

    lx, ly, lz = system.box_l
    particle_count = len(charges)
    positions = get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count
    )

    elc_errors = []
    contrib_data = {"P3M (3D)": [], "Yeh-Berkowitz": [], "ELC Reciprocal": []}

    for pw_err in pw_errors:
        system.part.clear()
        for i in range(particle_count):
            system.part.add(pos=positions[i], q=charges[i])

        ana_forces = get_ewald_forces_2d(system, n_max=100, prefactor=prefactor)

        pref, f_3d, f_elc_recip, f_corr_moments = get_elc_forces_contribs(
            system, gap_size, pw_err, prefactor
        )

        f_final = list(f_3d + pref * (f_elc_recip + f_corr_moments))

        i = 0  # choose a random particle
        elc_errors.append(
            abs(
                np.linalg.norm(f_final[i] - ana_forces[i])
                / np.linalg.norm(ana_forces[i])
            )
        )  # Vector L2 Relative Error

        contrib_data["P3M (3D)"].append(np.linalg.norm(f_3d[i]))
        contrib_data["Yeh-Berkowitz"].append(pref * np.linalg.norm(f_corr_moments[i]))
        contrib_data["ELC Reciprocal"].append(pref * np.linalg.norm(f_elc_recip[i]))

    # --- Assertions ---
    assert has_downward_trend(elc_errors)

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()

        colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
        bottoms = np.zeros(len(pw_errors))
        bar_width = 0.2 * np.array(pw_errors)

        # 1. Secondary Axis: Energy Contributions (Stacked Bars)
        for i, (label, vals) in enumerate(contrib_data.items()):
            ax2.bar(
                pw_errors,
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
            pw_errors,
            elc_errors,
            "o-",
            label="ELC Error",
            color="#2980b9",
            linewidth=2,
            zorder=5,
        )

        ax1.loglog(pw_errors, pw_errors, "k:", alpha=0.5, label="Target Accuracy (1:1)")

        # Formatting
        ax1.set_xlabel("Requested Accuracy (pw_error)")
        ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
        ax2.set_ylabel("Force Component Value (Linear Scale)", color="#7f8c8d")
        plt.title(title)

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
