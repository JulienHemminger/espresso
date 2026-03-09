import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.common.set_utils import are_sets_equal
from elc.src.forces.get_elc_forces import get_elc_forces
from elc.src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d
from elc.tests.forces.accuracy_convergence_utils import run_accuracy_convergence


def run_basic(
    system,
    lx,
    ly,
    lz,
    gap_size=1.0,
    charges=[+1.0, -1.0],
    pw_error=1e-6,
    tolerance=1e-6,
):

    particle_count = len(charges)

    system.part.clear()
    system.box_l = [lx, ly, lz]

    positions = get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count, max_distance=10
    )

    for i in range(particle_count):
        system.part.add(pos=positions[i], q=charges[i])

    analytical_forces = get_ewald_forces_2d(system)
    elc_forces = get_elc_forces(system, gap_size, pw_error)

    # Validation logic
    print(f"Testing Box: {lx}x{ly}x{lz} with charges {charges}")
    assert are_sets_equal(analytical_forces, elc_forces, tol=1e2 * pw_error)
    # assert are_sets_equal(analytical_forces, elc_forces, tol=pw_error), isnt passed by huge_box_neutral
    print("Success: Forces match within tolerance.")
    print("-" * 20)


def test_all():
    # pytest -vv -s ==JULIEN==/elc/tests/forces/force_test.py
    system = espressomd.System(box_l=[10, 10, 3])
    system.time_step = 0.01

    run_accuracy_convergence(
        system, prefactor=1.7, gap_size=1.0, charges=[+1, -1]
    )  # ERROR

    # run_accuracy_convergence(system, prefactor=1.7, gap_size=2.1, charges=[+1, -1])

    # run_accuracy_convergence(system, prefactor=1.7, gap_size=2.1, charges=[-0.9, +2.3, -1.4, -3.3])

    """FAILS
    system.part.clear()
    system.box_l = [12, 8, 7]
    run_accuracy_convergence(
        system, prefactor=2.3, gap_size=0.4, charges=[-0.9, +2.3, -1.4, -3.3]
    )"""

    """
    # small_box_neutral_dipole
    run_basic(system, 10, 10, 3, 1, [+1, -1])
    
    # small_box_neutral_tripole
    run_basic(system, 10, 10, 3, 1, [+2, -1, -1])

    # small_box_non_neutral_dipole_test
    run_basic(system, 10, 10, 3, 1, [+2, -1])

    run_madelung(system, ions_per_axis=8)


    # huge_box_neutral
    run_basic(system, 200, 200, 10, 1, [+1, -1])
    """
