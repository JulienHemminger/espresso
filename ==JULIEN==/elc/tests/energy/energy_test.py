import espressomd
import espressomd.electrostatics
from elc.src.common.get_charges import get_rdm_charges
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.energy.get_elc_energy import get_elc_energy
from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d


def run(
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
    elc_energy = get_elc_energy(system, gap_size, pw_error)

    # Validation logic
    print(f"Testing Box: {lx}x{ly}x{lz} with charges {charges}")
    assert abs(analytical_energy - elc_energy) <= 1e2 * pw_error
    print("Success: Energies match within tolerance.")
    print("-" * 20)


def test_all():
    system = espressomd.System(box_l=[1, 1, 1])
    system.time_step = 0.01
    """
    # small_box_neutral_dipole
    run(system, 10, 10, 3, 1, [+1, -1])

    # small_non_square_box_neutral_dipole
    run(system, 10, 7, 3, 1, [+1, -1])

    # small_box_neutral_tripole
    run(system, 10, 10, 3, 1, [+2, -1, -1])

    # small_box_non_neutral_dipole_test
    run(system, 10, 10, 3, 1, [+2, -1])

    # huge_box_neutral
    run(system, 200, 200, 10, 1, [+1, -1])
    """

    # varying gap_size
    # run(system, 10, 10, 3, gap_size=2, charges=[+1, -1])

    #
    run(system, 10, 10, 3, 1, charges=get_rdm_charges(2))

    # todo madelung, plots, more particles, ..
