import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.src.common.get_positions import get_rdm_constrained_points_np


def get_analytical_energy(system, params, pbc_images=10, reflection_steps=1):
    """
    Calculates energy considering both 2D periodic images (pbc_images)
    and dielectric reflection orders (reflection_steps).
    """
    pos = np.asarray(system.part.all().pos, dtype=np.float64)
    q = np.asarray(system.part.all().q, dtype=np.float64)

    lx, ly = params["lx"], params["ly"]
    gap = params["gap_size"]
    d_top = params["delta_mid_top"]
    d_bot = params["delta_mid_bot"]
    pref = params["prefactor"]

    N = len(q)
    E_total = 0.0

    # Setup 2D Periodic lattice vectors
    n_range = np.arange(-pbc_images, pbc_images + 1)
    NX, NY = np.meshgrid(n_range, n_range)
    Rx, Ry = (NX.ravel() * lx), (NY.ravel() * ly)
    origin_idx = np.where((Rx == 0) & (Ry == 0))[0][0]

    for i in range(N):
        for j in range(N):
            dx = pos[i, 0] - pos[j, 0] + Rx
            dy = pos[i, 1] - pos[j, 1] + Ry

            # --- Real-Real Interaction ---
            dz0 = pos[i, 2] - pos[j, 2]
            dist0 = np.sqrt(dx**2 + dy**2 + dz0**2)
            if i == j:
                dist0[origin_idx] = np.inf
            E_total += q[i] * q[j] * np.sum(1.0 / dist0)

            # --- Dielectric Reflection Summation ---
            for k in range(1, reflection_steps + 1):
                # Bottom interface reflections
                if d_bot != 0:
                    dz_bot = pos[i, 2] + pos[j, 2] + 2 * (k - 1) * gap
                    dist_bot = np.sqrt(dx**2 + dy**2 + dz_bot**2)
                    E_total += q[i] * q[j] * (d_bot**k) * np.sum(1.0 / dist_bot)

                # Top interface reflections
                if d_top != 0:
                    dz_top = 2 * gap - (pos[i, 2] + pos[j, 2]) + 2 * (k - 1) * gap
                    dist_top = np.sqrt(dx**2 + dy**2 + dz_top**2)
                    E_total += q[i] * q[j] * (d_top**k) * np.sum(1.0 / dist_top)

    return 0.5 * pref * E_total


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
    image_counts = [2**i for i in range(13)]  # Reduced range for faster calculation
    energies_pbc = [
        get_analytical_energy(system, params, pbc_images=n, reflection_steps=1)
        for n in image_counts
    ]

    # 2. Convergence over Reflection Steps (keeping PBC images constant)
    fixed_pbc = 64  # Use a value where PBC has sufficiently converged
    reflection_counts = list(range(1, 128))
    energies_refl = [
        get_analytical_energy(system, params, pbc_images=fixed_pbc, reflection_steps=k)
        for k in reflection_counts
    ]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
    ax2.set_title(f"Energy vs. Reflection Steps (at $n_max={fixed_pbc}$)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# System initialization
system = espressomd.System(box_l=[1.0, 1.0, 1.0])
system.time_step = 0.01

params = {
    "lx": 9.0,
    "ly": 12.0,
    "lz": 19.0,
    "gap_size": 15.0,
    "prefactor": 2.0,
    "delta_mid_top": 0.0,  # Adjusted to non-zero to see reflection effects
    "delta_mid_bot": -1.0,
    "charges": [+1, -1],
}
params["positions"] = get_rdm_constrained_points_np(
    params["lx"], params["ly"], params["lz"], len(params["charges"]), min_distance=0.1
)

run(system, **params, params=params)
