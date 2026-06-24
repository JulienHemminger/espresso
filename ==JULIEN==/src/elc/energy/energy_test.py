import espressomd
import espressomd.electrostatics
from common.generators.positions import get_rdm_constrained_points_np
from elc.energy.custom_elc_energy import get_elc_energy
from elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d
from elc.energy.accuracy_convergence_utils import run_accuracy_convergence
from elc.energy._3_madelung.madelung_utils import run_madelung


def run_basic(
    system, lx, ly, lz, gap_size=1.0, charges=[+1.0, -1.0], pw_error=1e-6, prefactor=1.0
):

    particle_count = len(charges)

    system.part.clear()
    system.box_l = [lx, ly, lz]

    positions = get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count, max_distance=10
    )

    for i in range(particle_count):
        system.part.add(pos=positions[i], q=charges[i])

    analytical_energy = get_ewald_energy_2d(system, prefactor=prefactor)
    elc_energy = get_elc_energy(system, gap_size, pw_error, prefactor=prefactor)

    # Validation logic
    print(f"Testing Box: {lx}x{ly}x{lz} with charges {charges}")
    assert abs(analytical_energy - elc_energy) <= 1e2 * pw_error
    print("Success: Energies match within tolerance.")
    print("-" * 20)


def test_all():
    # pytest -vv -s ==JULIEN==/elc/tests/energy/energy_test.py
    system = espressomd.System(box_l=[10, 10, 3])
    system.time_step = 0.01
    """
    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=1.0, charges=[+1, -1]
    )  # PASSED

    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=2.1, charges=[+1, -1]
    )  # PASSED

    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=2.1, charges=[-0.9, +1.1]
    )  # PASSED
    """
    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=2.1, charges=[-0.6, +1.5, -0.9]
    )  # PASSED
    """
    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=2.1, charges=[-0.9, +2.3, -1.4, -3.3]
    )  # PASSED

    system.part.clear()
    system.box_l = [7, 12, 2]
    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=2.1, charges=[-0.9, +2.3, -1.4, -3.3]
    )  # PASS

    # varying prefactor
    run_basic(system, 10, 7, 3, 1, [+1, -1], prefactor=1.7)
    run_basic(system, 10, 7, 3, 1, [+1, -1], prefactor=2.3)

    # small_box_neutral_dipole
    run_basic(system, 10, 10, 3, 1, [+1, -1])

    # small_non_square_box_neutral_dipole
    run_basic(system, 10, 7, 3, 1, [+1, -1])

    # small_box_neutral_tripole
    run_basic(system, 10, 10, 3, 1, [+2, -1, -1])

    # small_box_non_neutral_dipole_test
    run_basic(system, 10, 10, 3, 1, [+2, -1])

    # varying gap_size
    run_basic(system, 10, 10, 3, gap_size=2, charges=[+1, -1])

    run_madelung(system, ions_per_axis=8)

    # huge_box_neutral
    run_basic(system, 200, 200, 10, 1, [+1, -1])
    """

test_all()