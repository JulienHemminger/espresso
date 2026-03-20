import espressomd
import espressomd.electrostatics
import numpy as np
import pytest
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
from elcic.src.common.convergence_contribution_plot import (
    show_convergence_contribution_plot,
)
from elcic.src.energy.get_elcic_energy import get_elcic_energy_contribs
from elcic.src.energy.other.get_analytical_energy import calculate_elcic_energy


@pytest.fixture(scope="module")
def system():
    return espressomd.System(box_l=[1.0, 1.0, 1.0])


def setup_system(system, lx, ly, lz, gap_size, positions, charges):
    system.part.clear()
    system.box_l = [lx, ly, lz + gap_size]
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
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
    accuracies = [1e-5]
    energy_analytical, energy_legacy, energy_elcic = [], [], []
    e_3d_sums, e_corr_sums, e_far_vals = [], [], []

    setup_system(system, lx, ly, lz, gap_size, positions, charges)

    ana_energy = calculate_elcic_energy(system, params, n_max=2**9, k_max=2)

    for acc in accuracies:
        print(f"working on {acc=}")
        energy_analytical.append(ana_energy)

        energy_legacy.append(get_legacy_elc_energy(system, gap_size, acc))

        contribs = get_elcic_energy_contribs(
            system, gap_size, acc, prefactor, delta_mid_bot, delta_mid_top
        )
        e_3d_sums.append(sum(contribs[k]["e_3d"] for k in ["l0", "pm1", "lt"]))
        e_corr_sums.append(sum(contribs[k]["e_corr"] for k in ["l0", "pm1", "lt"]))
        e_far_vals.append(contribs["e_far"])
        energy_elcic.append(contribs["e_near"] + contribs["e_far"])

    show_convergence_contribution_plot(
        accuracies,
        e_3d_sums,
        e_corr_sums,
        e_far_vals,
        energy_analytical,
        energy_legacy,
        energy_elcic,
        params,
    )


def test_all(system):
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
        params["lx"],
        params["ly"],
        params["lz"],
        len(params["charges"]),
        min_distance=0.1,
    )
    params["positions"] = [np.array([7, 1, 3]), np.array([4, 5, 2])]

    run(system, **params, params=params)
