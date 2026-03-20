import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.src.common.get_positions import get_rdm_constrained_points_np


def get_analytical_energy(system, params, n_max=10):
    positions = np.array([p.pos for p in system.part])  # Shape (N, 3)
    charges = np.array([p.q for p in system.part])  # Shape (N,)
    N = len(charges)
    lz = params["lz"]
    prefactor = params["prefactor"]
    delta_b = params["delta_mid_bot"]
    delta_t = params["delta_mid_top"]
    delta = delta_b * delta_t

    energy = 0.0

    # 1. Real-Real interactions (Standard Coulomb)
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            energy += prefactor * (charges[i] * charges[j]) / r

    # 2. Real-Image interactions
    # We iterate through generations n=0 to n_max for each of the 4 image sequences
    for i in range(N):
        zi = positions[i][2]
        for j in range(N):
            zj = positions[j][2]
            qi_qj = charges[i] * charges[j]
            # Horizontal distance squared (x and y components)
            r_dist_sq = np.sum((positions[i][:2] - positions[j][:2]) ** 2)

            for n in range(n_max + 1):
                # Lower sequence 1 (Eq 2.2): Charge qj * delta_b * delta^n at -(2*n*lz + zj)
                z_img_l1 = -(2 * n * lz + zj)
                energy += (
                    0.5
                    * prefactor
                    * (qi_qj * delta_b * (delta**n))
                    / np.sqrt(r_dist_sq + (zi - z_img_l1) ** 2)
                )

                # Upper sequence 1 (Eq 2.4): Charge qj * delta_t * delta^n at (2*(n+1)*lz - zj)
                z_img_u1 = 2 * (n + 1) * lz - zj
                energy += (
                    0.5
                    * prefactor
                    * (qi_qj * delta_t * (delta**n))
                    / np.sqrt(r_dist_sq + (zi - z_img_u1) ** 2)
                )

                if n > 0:
                    # Lower sequence 2 (Eq 2.3): Charge qj * delta^n at -(2*n*lz - zj)
                    z_img_l2 = -(2 * n * lz - zj)
                    energy += (
                        0.5
                        * prefactor
                        * (qi_qj * (delta**n))
                        / np.sqrt(r_dist_sq + (zi - z_img_l2) ** 2)
                    )

                    # Upper sequence 2 (Eq 2.5): Charge qj * delta^n at (2*n*lz + zj)
                    z_img_u2 = 2 * n * lz + zj
                    energy += (
                        0.5
                        * prefactor
                        * (qi_qj * (delta**n))
                        / np.sqrt(r_dist_sq + (zi - z_img_u2) ** 2)
                    )

    return energy


def setup_system(system, lx, ly, lz, gap_size, positions, charges):
    system.part.clear()
    system.box_l = [lx, ly, lz + gap_size]
    system.time_step = 0.01
    for i in range(len(charges)):
        system.part.add(pos=positions[i], q=charges[i])
    return system


import pprint


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
    image_counts = [2**i for i in range(8)]  # Reduced range for faster calculation
    energies_pbc = [
        get_analytical_energy(system, params, n_max=n) for n in image_counts
    ]

    # 2. Convergence over Reflection Steps (keeping PBC images constant)
    fixed_pbc = 64  # Use a value where PBC has sufficiently converged
    reflection_counts = list(range(1, 16))
    energies_refl = [
        get_analytical_energy(system, params, n_max=k) for k in reflection_counts
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
    ax2.set_title(f"Energy vs. Reflection Steps (at n_max={fixed_pbc})")
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
    "delta_mid_top": 0.0,  # Adjusted to non-zero to see reflection effects
    "delta_mid_bot": -1.0,
    "charges": [+1, -1],
}
params["positions"] = get_rdm_constrained_points_np(
    params["lx"], params["ly"], params["lz"], len(params["charges"]), min_distance=0.1
)
params["positions"] = [np.array([7, 1, 3]), np.array([4, 5, 2])]

run(system, **params, params=params)


# -0.263
