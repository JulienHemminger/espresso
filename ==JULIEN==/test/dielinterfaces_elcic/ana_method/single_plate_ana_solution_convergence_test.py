import pprint

import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from src.common.has_downward_trend import has_downward_trend
from src.energy.third_party.analytical_elcic_energy import analytical_elcic_energy


@pytest.fixture
def system():
    """Fixture to manage the ESPResSo system lifecycle."""
    sys = espressomd.System(box_l=[1.0, 1.0, 1.0])
    sys.time_step = 0.01
    yield sys
    sys.part.clear()


def test_energy_convergence(system):
    # Parameters
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
    positions = [np.array([7, 1, 3]), np.array([4, 5, 2])]

    # Setup System
    system.box_l = [params["lx"], params["ly"], params["lz"] + params["gap_size"]]
    for i, q in enumerate(params["charges"]):
        system.part.add(pos=positions[i], q=q)

    energies_pbc = []
    energies_refl = []
    N = 9
    image_counts = [2**i for i in range(N)]
    reflection_counts = list(range(N))

    for i in range(N):
        energies_pbc.append(analytical_elcic_energy(system, params, n_max=2**i))
        energies_refl.append(analytical_elcic_energy(system, params, k_max=i))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(
        image_counts, energies_pbc, "o-", color="tab:blue", label="PBC Convergence"
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Number of PBC Images ($n_{max}$)")
    ax1.set_ylabel(r"Energy ($U_{total}$)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(
        reflection_counts,
        energies_refl,
        "s-",
        color="tab:red",
        label="Reflection Convergence",
    )
    ax2.set_xlabel(r"Reflection Steps ($k_{max}$)")
    ax2.set_ylabel(r"Energy ($U_{total}$)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout(rect=(0, 0.15, 1, 1))
    stats_text = f"Diff: {energies_pbc[-1] - energies_refl[-1]:.6e}\nParams: {pprint.pformat(params)}"
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.show()

    max_limit_diff = 9e-3
    # Detailed status report
    print("--- Energy Comparison Report ---")
    print(f"energies_pbc={[round(float(e), 7) for e in energies_pbc]}")
    print(f"energies_refl={[round(float(e), 7) for e in energies_refl]}")
    print(
        f"Converge Towards Same Limit: {has_downward_trend(np.array(energies_pbc) - np.array(energies_refl))}"
    )
    print(
        f"Limit Difference: {np.abs(energies_refl[-1] - energies_pbc[-1])} <= {max_limit_diff=}: {np.abs(energies_refl[-1] - energies_pbc[-1]) <= max_limit_diff}"
    )
    print("--------------------------------")

    # Assertions
    # assert has_downward_trend(np.array(energies_pbc) - np.array(energies_refl))
    assert np.abs(energies_refl[-1] - energies_pbc[-1]) <= max_limit_diff
