import matplotlib.pyplot as plt
import numpy as np


def show_convergence_contribution_plot(
    accuracies, e_3d_sums, e_corr_sums, e_far_vals, errors_legacy, errors_elcic
):
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

    plt.title("ELCIC Convergence & Energy Breakdown)")
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
