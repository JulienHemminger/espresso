import espressomd
import espressomd.electrostatics
import numpy as np
import pytest
from elcic.src.common.convergence_contribution_plot import (
    show_convergence_contribution_plot,
)
from elcic.src.energy.get_analytical_energy import calculate_elcic_energy
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs


@pytest.fixture(scope="module")
def system():
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, lx, ly, lz, gap_size, charges):
    system.part.clear()
    system.box_l = [lx, ly, lz + gap_size]
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
    system.time_step = 0.01

    for q in charges:
        # Generate random position within [0, lx], [0, ly], [0, lz]
        # Ensuring 0.1 minimum distance between particles
        while True:
            pos = np.random.rand(3) * [lx, ly, lz]
            if len(system.part) == 0:
                break
            dist = np.linalg.norm(system.part.all().pos - pos, axis=1)
            if np.all(dist > 0.1):
                break
        system.part.add(pos=pos, q=q)
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
    charges,
    params,
):
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    errors_legacy, errors_elcic = [], []
    e_3d_sums, e_corr_sums, e_far_vals = [], [], []

    setup_system(system, lx, ly, lz, gap_size, charges)
    params["positions"] = system.part.all().pos
    ana_energy = calculate_elcic_energy(params)

    for acc in accuracies:
        # 1. Legacy ELC
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=acc, check_neutrality=False
        )
        elc = espressomd.electrostatics.ELC(
            actor=p3m,
            gap_size=gap_size,
            maxPWerror=acc,
            delta_mid_bot=delta_mid_bot,
            delta_mid_top=delta_mid_top,
            neutralize=False,
        )
        system.electrostatics.solver = elc
        errors_legacy.append(abs(system.analysis.energy()["total"] - ana_energy))

        # 2. ELCIC Decomposition
        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )
        e_3d_sums.append(sum(contribs[k]["e_3d"] for k in ["l0", "pm1", "lt"]))
        e_corr_sums.append(sum(contribs[k]["e_corr"] for k in ["l0", "pm1", "lt"]))
        e_far_vals.append(contribs["e_far"])
        errors_elcic.append(abs((contribs["e_near"] + contribs["e_far"]) - ana_energy))

    show_convergence_contribution_plot(
        accuracies, e_3d_sums, e_corr_sums, e_far_vals, errors_legacy, errors_elcic
    )


def test_all(system):
    params = {
        "lx": 20.0,
        "ly": 20.0,
        "lz": 20.0,
        "gap_size": 15.0,
        "prefactor": 2.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": 0.95,
        "charges": [+1, -1, +1, -1],  # Supports arbitrary charge lists
    }
    run(system, **params, params=params)

    """
    # SINGLE PLATE
    # run(system, **params)  # neutral, metallic, PASS

    params["delta_mid_bot"] = 0.9
    # run(system, **params)  # neutral, non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    # run(system, **params)  # non-neutral, non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, metallic, FAIL - legacy elc doesnt work

    # DOUBLE PLATES
    params["charges"] = [+1, -1]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # neutral, both metallic, PASS

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    # run(system, **params)  # neutral, both non-metallic, PASS

    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # neutral, mixed metallic + non-metallic, PASS

    params["charges"] = [+1.2, -0.7]
    params["delta_mid_top"] = -1.0
    params["delta_mid_bot"] = -1.0
    # run(system, **params)  # non-neutral, both metallic, FAIL - legacy elc doesnt work

    params["delta_mid_top"] = 0.7
    params["delta_mid_bot"] = 0.7
    # run(system, **params)  # non-neutral, both non-metallic, FAIL - legacy elc doesnt work

    params["delta_mid_bot"] = -1.0
    # run_test(system, **params)  # non-neutral, mixed metallic + non-metallic, FAIL - legacy elc doesnt work
    """
