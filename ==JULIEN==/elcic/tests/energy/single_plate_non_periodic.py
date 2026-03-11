import espressomd
import espressomd.electrostatics
import numpy as np
from elcic.src.energy.get_elcic_energy import get_elcic_energy


def setup_system(
    box_l,
    gap_size,
    p1_pos_z,
    r_p1_p2,
    prefactor,
    accuracy,
    delta_mid_bot,
    delta_mid_top,
    charges=[+1, -1],
):
    """Initializes the ESPResSo system with two fixed particles."""
    half_box_l = box_l / 2.0

    # Define the system with the gap included in the z-dimension
    system = espressomd.System(box_l=[box_l, box_l, box_l + gap_size])
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    # Add particles with q1=+1 and q2=-1 at fixed z-positions
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z], q=charges[0])
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z + r_p1_p2], q=charges[1])

    return system


def calculate_analytic(z, dist, prefactor, delta_mid_bot):
    """Calculates analytic force and energy for q=1 at a specific z."""
    # Force: F = q^2 * prefactor * (1/d^2 + delta * (1/(2z)^2 - 1/(2z+d)^2))
    force = prefactor * (
        1 / dist**2 + delta_mid_bot * (1 / (2 * z) ** 2 - 1 / (2 * z + dist) ** 2)
    )

    # Energy: Interaction energy of dipole + interaction with images
    energy = prefactor * (
        -1 / dist
        + delta_mid_bot * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )
    return force, energy


def run_test(
    box_l,
    gap_size,
    accuracy,
    prefactor,
    p1_pos_z,
    r_p1_p2,
    delta_mid_top,
    delta_mid_bot,
    charges=[+1, -1],
):
    """Executes the simulation and compares against analytic results."""

    system = setup_system(
        box_l,
        gap_size,
        p1_pos_z,
        r_p1_p2,
        prefactor,
        accuracy,
        delta_mid_bot,
        delta_mid_top,
        charges,
    )

    # Legacy ELC

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False
    )
    elc = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=gap_size,
        maxPWerror=accuracy,
        delta_mid_bot=delta_mid_bot,
        delta_mid_top=delta_mid_top,
        neutralize=False,
        check_neutrality=False,
    )
    system.electrostatics.solver = elc
    system.integrator.run(0)  # Update forces
    p1 = system.part.by_id(0)
    elc_force = p1.f[2]

    elc_energy = get_elcic_energy(
        system,
        gap_size,
        accuracy,
        prefactor,
        delta_mid_bot=delta_mid_bot,
        delta_mid_top=delta_mid_top,
    )
    elc_force = 0

    # Get Analytic results
    ana_force, ana_energy = calculate_analytic(
        p1_pos_z, r_p1_p2, prefactor, delta_mid_bot
    )

    # Output results
    print(f"--- Comparison at z={p1_pos_z} ---")
    print(
        f"Force  | ELC: {elc_force:10.7f} | Analytic: {ana_force:10.7f} | Diff: {elc_force - ana_force:.2e}"
    )
    print(
        f"Energy | ELC: {elc_energy:10.7f} | Analytic: {ana_energy:10.7f} | Diff: {elc_energy - ana_energy:.2e}"
    )

    # Final Validation
    np.testing.assert_allclose(elc_force, ana_force, atol=1e-4)
    np.testing.assert_allclose(elc_energy, ana_energy, atol=1e-4)
    print("Verification successful.")


if __name__ == "__main__":
    params = {
        "box_l": 200.0,
        "gap_size": 75.0,
        "accuracy": 1e-6,
        "prefactor": 2.0,
        "p1_pos_z": 10.0,
        "r_p1_p2": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "charges": [+1, -1],
    }
    # run_test(**params)  # neutral, metallic, PASS

    params["delta_mid_bot"] = 0.9
    # run_test(**params)  # neutral, non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    # run_test(**params)  # non-neutral, non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(**params)  # non-neutral, metallic, FAIL - legacy elc doesnt work
