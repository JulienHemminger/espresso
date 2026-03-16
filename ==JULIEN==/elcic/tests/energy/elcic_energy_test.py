import espressomd
import espressomd.electrostatics
import pytest
from elcic.src.common.convergence_contribution_plot import (
    show_convergence_contribution_plot,
)
from elcic.src.energy.get_analytical_energy import calculate_elcic_energy
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs


@pytest.fixture(scope="module")
def system():
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges=[+1, -1]):
    system.part.clear()
    system.box_l = [box_l, box_l, box_l + gap_size]
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
    system.time_step = 0.01

    half_box_l = box_l / 2.0
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z], q=charges[0])
    system.part.add(pos=[half_box_l, half_box_l, p1_pos_z + r_p1_p2], q=charges[1])
    return system


def run(
    system,
    box_l,
    gap_size,
    prefactor,
    p1_pos_z,
    r_p1_p2,
    delta_mid_top,
    delta_mid_bot,
    charges=[+1, -1],
    params={},
):
    accuracies = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    errors_legacy = []
    errors_elcic = []

    # Storage for stacked bars
    e_3d_sums = []
    e_corr_sums = []
    e_far_vals = []

    setup_system(system, box_l, gap_size, p1_pos_z, r_p1_p2, charges)
    ana_energy = calculate_elcic_energy(params)

    for acc in accuracies:
        # 1. Legacy ELC Calculation
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
        legacy_energy = system.analysis.energy()["total"]
        errors_legacy.append(abs(legacy_energy - ana_energy))

        # 2. ELCIC Decomposition Calculation
        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )

        # Aggregate components for the bar chart
        e_3d_sums.append(
            contribs["l0"]["e_3d"] + contribs["pm1"]["e_3d"] + contribs["lt"]["e_3d"]
        )
        e_corr_sums.append(
            contribs["l0"]["e_corr"]
            + contribs["pm1"]["e_corr"]
            + contribs["lt"]["e_corr"]
        )
        e_far_vals.append(contribs["e_far"])

        # Calculate ELCIC specific error
        elcic_total = contribs["e_near"] + contribs["e_far"]
        errors_elcic.append(abs(elcic_total - ana_energy))

    show_convergence_contribution_plot(
        accuracies, e_3d_sums, e_corr_sums, e_far_vals, errors_legacy, errors_elcic
    )


def test_all(system):
    params = {
        "box_l": 20.0,  # default=200, PASS=[], FAIL=[]
        "gap_size": 15.0,  # default=75, PASS=[15, 20], FAIL=[5, 10]
        "prefactor": 2.0,
        "p1_pos_z": 1.0,  # default=10, PASS=[], FAIL=[]
        "r_p1_p2": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": 0.95,
        "charges": [+1, -1],
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
