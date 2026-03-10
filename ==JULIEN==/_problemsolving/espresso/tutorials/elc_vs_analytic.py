import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & Constants ---
BOX_L = 200.0
ELC_GAP = 75.0
ACCURACY = 1e-7
RTOL = 1e-7
ATOL = 1e-4
PREFACTOR = 2.0
DISTANCE = 1.0

DELTA_MID_TOP = 0.0
DELTA_MID_BOT = 39.0 / 41.0

# Setup z-positions
MIN_DIST_WALL = 0.1
num_points = 6 if espressomd.code_info.build_type() == "Coverage" else 12
Z_POSITIONS = np.linspace(MIN_DIST_WALL, BOX_L - MIN_DIST_WALL - DISTANCE, num_points)
CHARGES = np.arange(-5.0, 5.1, 2.5)


def setup_system():
    """Initializes the ESPResSo system and electrostatic solver."""
    system = espressomd.System(box_l=[BOX_L, BOX_L, BOX_L + ELC_GAP])
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    # Initialize with the first non-zero charge to satisfy P3M's validation
    initial_q = CHARGES[0] if CHARGES[0] != 0 else CHARGES[1]

    center = BOX_L / 2.0
    system.part.add(pos=[center, center, Z_POSITIONS[0]], q=initial_q)
    system.part.add(pos=[center, center, Z_POSITIONS[0] + DISTANCE], q=-initial_q)

    p3m = espressomd.electrostatics.P3M(
        prefactor=PREFACTOR, accuracy=ACCURACY, mesh=[58, 58, 70], cao=4, gpu=False
    )
    elc = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=ELC_GAP,
        maxPWerror=ACCURACY,
        delta_mid_bot=DELTA_MID_BOT,
        delta_mid_top=DELTA_MID_TOP,
    )

    # Now this assignment will succeed because charges exist
    system.electrostatics.solver = elc
    return system


def get_analytic_results(q_array, z_array):
    """Calculates analytic force and energy using image charge method."""
    q_sq = PREFACTOR * np.square(q_array.reshape(-1, 1))

    forces = q_sq * (
        1 / DISTANCE**2
        + DELTA_MID_BOT * (1 / (2 * z_array) ** 2 - 1 / (2 * z_array + DISTANCE) ** 2)
    )

    energy = q_sq * (
        -1 / DISTANCE
        + DELTA_MID_BOT
        * (
            1 / (4 * z_array)
            - 1 / (2 * z_array + DISTANCE)
            + 1 / (4 * (z_array + DISTANCE))
        )
    )
    return forces, energy


def get_numerical_results(system):
    """Iterates through charges and positions to gather ELC data."""
    # Note: particle 0 is p1, particle 1 is p2
    p1, p2 = system.part.all()
    elc_forces = np.empty((len(CHARGES), len(Z_POSITIONS)))
    elc_energy = np.empty(elc_forces.shape)

    for i, q in enumerate(CHARGES):
        p1.q, p2.q = q, -q
        for j, z in enumerate(Z_POSITIONS):
            p1.pos = [BOX_L / 2, BOX_L / 2, z]
            p2.pos = [BOX_L / 2, BOX_L / 2, z + DISTANCE]

            system.integrator.run(0)
            elc_forces[i, j] = p1.f[2]
            elc_energy[i, j] = system.analysis.energy()["total"]

    return elc_forces, elc_energy


def plot_comparison(z_vals, sim_f, ana_f, sim_e, ana_e):
    """Plots raw data comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting first non-zero charge index (usually index 0: -5.0)
    idx = 0
    ax1.plot(z_vals, ana_f[idx], "k-", label="Analytic", alpha=0.6)
    ax1.scatter(z_vals, sim_f[idx], color="red", label="ELC (Raw)", s=30)
    ax1.set_title(f"Force Comparison (q={CHARGES[idx]})")
    ax1.set_xlabel("z Position")
    ax1.set_ylabel("Force $F_z$")
    ax1.legend()

    ax2.plot(z_vals, ana_e[idx], "k-", label="Analytic", alpha=0.6)
    ax2.scatter(z_vals, sim_e[idx], color="blue", label="ELC (Raw)", marker="s", s=30)
    ax2.set_title(f"Energy Comparison (q={CHARGES[idx]})")
    ax2.set_xlabel("z Position")
    ax2.set_ylabel("Total Energy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    system = setup_system()

    num_f, num_e = get_numerical_results(system)
    ana_f, ana_e = get_analytic_results(CHARGES, Z_POSITIONS)

    # Validation
    np.testing.assert_allclose(num_e, ana_e, atol=ATOL)
    np.testing.assert_allclose(num_f, ana_f, atol=ATOL, rtol=RTOL)

    plot_comparison(Z_POSITIONS, num_f, ana_f, num_e, ana_e)
