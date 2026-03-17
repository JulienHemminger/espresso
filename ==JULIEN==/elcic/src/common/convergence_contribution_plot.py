import matplotlib.pyplot as plt
import numpy as np


def show_convergence_contribution_plot(
    accuracies,
    e_3d_sums,
    e_corr_sums,
    e_far_vals,
    energy_analytical,
    energy_legacy,
    energy_elcic,
    params={"test": True},
):
    a_3d = np.abs(e_3d_sums)
    a_corr = np.abs(e_corr_sums)
    a_far = np.abs(e_far_vals)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    plt.subplots_adjust(left=0.2)
    x_pos = np.arange(len(accuracies))

    # Bar Chart
    ax1.bar(x_pos, a_3d, label="|Sum E_3D|", alpha=0.3, color="blue")
    ax1.bar(x_pos, a_corr, bottom=a_3d, label="|Sum E_Corr|", alpha=0.3, color="green")
    ax1.bar(
        x_pos, a_far, bottom=a_3d + a_corr, label="|E_Far|", alpha=0.3, color="orange"
    )

    # Parameter Text Box (kept same)
    param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
    ax1.text(
        -0.22,
        0.5,
        f"Parameters:\n{param_str}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    ax1.set_ylabel("Energy Component Magnitude", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{a:.0e}" for a in accuracies])
    ax1.legend(loc="upper left")

    # Scatter Plots for Actual Values
    ax2 = ax1.twinx()
    ax2.plot(
        x_pos,
        energy_analytical,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Analytical",
    )
    ax2.scatter(
        x_pos, energy_legacy, color="black", marker="D", s=80, label="Legacy Value"
    )
    ax2.scatter(x_pos, energy_elcic, color="red", marker="o", s=80, label="ELCIC Value")

    ax2.set_ylabel("Total Energy Value")
    # Log scale is removed for ax2 as actual values may be negative or cross zero
    ax2.legend(loc="upper right")

    plt.title("ELCIC Convergence & Energy Value Comparison")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    # Print original signed values
    print("\n" + "=" * 50)
    print(f"{'Accuracy':<10} | {'Sum E_3D':<12} | {'Sum E_Corr':<12} | {'E_Far':<12}")
    print("-" * 50)
    for i, acc in enumerate(accuracies):
        print(
            f"{acc:<10.0e} | {e_3d_sums[i]:<12.6f} | {e_corr_sums[i]:<12.6f} | {e_far_vals[i]:<12.6f}"
        )
    print("=" * 50 + "\n")
