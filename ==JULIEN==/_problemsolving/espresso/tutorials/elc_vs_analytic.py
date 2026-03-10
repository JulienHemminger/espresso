import espressomd
import espressomd.electrostatics
import numpy as np

# --- Configuration & Constants ---
BOX_L = 200.0
GAP_SIZE = 75.0
ACCURACY = 1e-7
PREFACTOR = 2.0

# Fixed Positions
Z_POS = 10.0  # z-position of the first particle
CENTER = BOX_L / 2.0
DISTANCE = 1.0  # Vertical distance between particles

# Dielectric contrast
DELTA_MID_TOP = 0.0
DELTA_MID_BOT = 39.0 / 41.0


def setup_system():
    """Initializes the ESPResSo system with two fixed particles."""
    system = espressomd.System(box_l=[BOX_L, BOX_L, BOX_L + GAP_SIZE])
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    # Add particles with q1=+1 and q2=-1 at fixed z-positions
    system.part.add(pos=[CENTER, CENTER, Z_POS], q=1.0)
    system.part.add(pos=[CENTER, CENTER, Z_POS + DISTANCE], q=-1.0)

    p3m = espressomd.electrostatics.P3M(prefactor=PREFACTOR, accuracy=ACCURACY)
    elc = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=GAP_SIZE,
        maxPWerror=ACCURACY,
        delta_mid_bot=DELTA_MID_BOT,
        delta_mid_top=DELTA_MID_TOP,
    )
    system.electrostatics.solver = elc
    return system


def calculate_analytic(z, dist):
    """Calculates analytic force and energy for q=1 at a specific z."""
    # Based on image charge method for a dipole near a dielectric interface
    # F = q^2 * prefactor * (1/d^2 + delta * (1/(2z)^2 - 1/(2z+d)^2))
    force = PREFACTOR * (
        1 / dist**2 + DELTA_MID_BOT * (1 / (2 * z) ** 2 - 1 / (2 * z + dist) ** 2)
    )

    energy = PREFACTOR * (
        -1 / dist
        + DELTA_MID_BOT * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )
    return force, energy


if __name__ == "__main__":
    system = setup_system()
    system.integrator.run(0)  # Update forces

    # Get Simulation results
    p1 = system.part.by_id(0)
    elc_force = p1.f[2]
    elc_energy = system.analysis.energy()["total"]

    # Get Analytic results
    ana_force, ana_energy = calculate_analytic(Z_POS, DISTANCE)

    # Output results
    print(f"--- Comparison at z={Z_POS} ---")
    print(
        f"Force  | ELC: {elc_force:10.7f} | Analytic: {ana_force:10.7f} | Diff: {elc_force - ana_force:.2e}"
    )
    print(
        f"Energy | ELC: {elc_energy:10.7f} | Analytic: {ana_energy:10.7f} | Diff: {elc_energy - ana_energy:.2e}"
    )

    # Final Validation
    np.testing.assert_allclose(elc_force, ana_force, atol=1e-4)
    np.testing.assert_allclose(elc_energy, ana_energy, atol=1e-4)
    print("\nVerification successful.")
