import pprint

import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.common.get_positions import get_rdm_constrained_points_np
from src.energy.third_party.analytical_elcic_energy import analytical_elcic_energy


def setup_system(system, lx, ly, lz, gap_size, positions, charges):
    system.part.clear()
    system.box_l = [lx, ly, lz + gap_size]
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

    # 1. Convergence over PBC Images (keeping reflections constant)
    image_counts = [2**i for i in range(2)]  # Reduced range for faster calculation
    energies_pbc = [
        analytical_elcic_energy(system, params, n_max=n) for n in image_counts
    ]

    # 2. Convergence over Reflection Steps (keeping PBC images constant)
    reflection_counts = [2, 4]
    energies_refl = [
        analytical_elcic_energy(system, params, k_max=k) for k in reflection_counts
    ]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6)
    )  # Increased height slightly for label space

    # Plot 1: PBC Image Convergence
    ax1.plot(
        image_counts, energies_pbc, "o-", color="tab:blue", label="PBC Convergence"
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(image_counts)
    ax1.set_xlabel(r"Number of PBC Images ($n_{max}$)")
    ax1.set_ylabel(r"Total Electrostatic Energy ($U_{total}$)")
    ax1.set_title("Energy vs. PBC Images")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Plot 2: Reflection Step Convergence
    ax2.plot(
        reflection_counts,
        energies_refl,
        "s-",
        color="tab:red",
        label="Reflection Convergence",
    )
    ax2.set_xlabel(r"Reflection Steps ($k_{max}$)")
    ax2.set_ylabel(r"Total Electrostatic Energy ($U_{total}$)")
    ax2.set_title("Energy vs. Reflection Steps")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.subplots_adjust(bottom=0.3, top=0.9, wspace=0.3)

    params_str = pprint.pformat(params, indent=4)
    stats_text = (
        f"Energy PBC: {energies_pbc[-1]:.6f} | Energy Refl: {energies_refl[-1]:.6f} | "
        f"Diff: {energies_pbc[-1] - energies_refl[-1]:.6e}\n"
        f"Params: {params_str}"
    )

    # Place text at y=0.02 (inside the figure, but below the axes)
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.show()


# System initialization
system = espressomd.System(box_l=[1.0, 1.0, 1.0])
system.time_step = 0.01

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
params["positions"] = get_rdm_constrained_points_np(
    params["lx"], params["ly"], params["lz"], len(params["charges"]), min_distance=0.1
)
params["positions"] = [np.array([7, 1, 3]), np.array([4, 5, 2])]

run(system, **params, params=params)
