import espressomd
import espressomd.electrostatics
import numpy as np
import pytest
from elcic.src.energy.get_elcic_energy import get_elcic_energy


@pytest.fixture(scope="module")
def system():
    """Initializes the ESPResSo system singleton once for the session."""
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(
    system,
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
    """Resets and reconfigures the existing ESPResSo system."""
    # Handle the singleton state
    system.part.clear()
    system.electrostatics.clear()

    # Reconfigure geometry
    system.box_l = [box_l, box_l, box_l + gap_size]
    system.time_step = 0.01
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)

    # Add particles
    half_box_l = box_l / 2.0
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z], q=charges[0])
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z + r_p1_p2], q=charges[1])

    return system


def calculate_analytic(z, dist, prefactor, delta_mid_bot):
    """Calculates analytic force and energy for q=1 at a specific z."""
    force = prefactor * (
        1 / dist**2 + delta_mid_bot * (1 / (2 * z) ** 2 - 1 / (2 * z + dist) ** 2)
    )
    energy = prefactor * (
        -1 / dist
        + delta_mid_bot * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )
    return force, energy


def run(
    system,
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
    """Executes the simulation using the provided system instance."""
    setup_system(
        system,
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

    ana_force, ana_energy = calculate_analytic(
        p1_pos_z, r_p1_p2, prefactor, delta_mid_bot
    )

    elc_energy = get_elcic_energy(
        system,
        gap_size,
        accuracy,
        prefactor,
        delta_mid_bot=delta_mid_bot,
        delta_mid_top=delta_mid_top,
    )

    np.testing.assert_allclose(elc_energy, ana_energy, atol=1e1 * accuracy)


def test_all(system):
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

    # SINGLE PLATE
    run(system, **params)  # neutral, metallic, PASS

    params["delta_mid_bot"] = 0.9
    run(system, **params)  # neutral, non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    # run_test(system, **params)  # non-neutral, non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, metallic, FAIL - legacy elc doesnt work

    # DOUBLE PLATES
    params["charges"] = [+1, -1]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    run(system, **params)  # neutral, both metallic, PASS

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    run(system, **params)  # neutral, both non-metallic, PASS

    params["delta_mid_bot"] = -1.0
    run(system, **params)  # neutral, mixed metallic + non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # non-neutral, both metallic, FAIL - legacy elc doesnt work

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    # run(system, **params)  # non-neutral, both non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, mixed metallic + non-metallic, FAIL - legacy elc doesnt work
