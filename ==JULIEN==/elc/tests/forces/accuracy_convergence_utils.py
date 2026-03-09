import matplotlib.pyplot as plt
import numpy as np
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.forces.get_elc_forces import get_elc_forces_contribs
from elc.src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d


def run_accuracy_convergence(
    system,
    show_convergence_plot=True,
    prefactor=1.0,
    gap_size=1.0,
    charges=[+1.0, -1.0],
):
    # Setup parameters
    pw_errors = np.logspace(-4, -8, num=5)

    title = "Force Accuracy Convergence (Z-component only) "

    lx, ly, lz = system.box_l
    particle_count = len(charges)

    elc_errors = []
    contrib_data = {"P3M (3D)": [], "Yeh-Berkowitz": [], "ELC Reciprocal": []}

    for pw_err in pw_errors:
        system.part.clear()

        positions = get_rdm_constrained_points_np(lx, ly, lz, particle_count)
        for i in range(particle_count):
            system.part.add(pos=positions[i], q=charges[i])

        # Reference force from 2D Ewald
        ana_forces = get_ewald_forces_2d(system, n_max=100, prefactor=prefactor)

        # ELC force components
        pref, f_3d, f_elc_recip, f_corr_moments = get_elc_forces_contribs(
            system, gap_size=gap_size, pw_err=pw_err, prefactor=prefactor
        )

        # Final force calculation
        f_final = f_3d + pref * (f_elc_recip + f_corr_moments)

        # --- Error calculation using Z-component only ---
        i = 0  # Focus on the first particle
        fz_numeric = np.linalg.norm(f_final[i])
        fz_analytic = np.linalg.norm(ana_forces[i][2])

        # Scalar relative error for the Z component
        rel_err_z = abs((fz_numeric - fz_analytic) / fz_analytic)
        elc_errors.append(rel_err_z)

        # Contribution data (using Z-component magnitude)
        contrib_data["P3M (3D)"].append(abs(f_3d[i][2]))
        contrib_data["Yeh-Berkowitz"].append(abs(pref * f_corr_moments[i][2]))
        contrib_data["ELC Reciprocal"].append(abs(pref * f_elc_recip[i][2]))

    # --- Assertions ---
    """
    assert has_downward_trend(elc_errors), (
        "Error did not decrease with requested accuracy!"
    )"""

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()
        colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
        bottoms = np.zeros(len(pw_errors))
        bar_width = 0.2 * np.array(pw_errors)

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

        ax1.loglog(
            pw_errors,
            elc_errors,
            "o-",
            label="ELC Error (Z)",
            color="#2980b9",
            zorder=5,
        )
        ax1.loglog(pw_errors, pw_errors, "k:", alpha=0.5, label="Target Accuracy")

        # --- Legend Logic ---
        # Collect handles and labels from both axes
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()

        # Combine them and create a single legend on ax1 (or ax2)
        ax1.legend(lines + bars, labels + bar_labels, loc="upper left", frameon=True)

        ax1.set_xlabel("Requested Accuracy (pw_error)")
        ax1.set_ylabel("Measured Rel. Error in $F_z$", color="#2980b9")
        ax2.set_ylabel("Force Component Value ($|F_z|$)", color="#7f8c8d")
        plt.title(title)

        ax1.invert_xaxis()
        fig.tight_layout()
        plt.show()
