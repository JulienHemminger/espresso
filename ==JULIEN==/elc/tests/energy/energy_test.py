import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.energy.get_elc_energy import get_elc_energy_new
from elc.src.energy.third_party.get_legacy_elc import (
    get_legacy_elc_energy_new,
)


def run(
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

    analytical_energy = get_legacy_elc_energy_new(system, gap_size, pw_error)
    elc_energy = get_elc_energy_new(system, gap_size, pw_error)

    # Validation logic
    print(f"Testing Box: {lx}x{ly}x{lz} with charges {charges}")
    assert abs(analytical_energy - elc_energy) <= 1e2 * pw_error
    print("Success: Energies match within tolerance.")
    print("-" * 20)


def test_all():
    system = espressomd.System(box_l=[1, 1, 1])
    system.time_step = 0.01

    # dipole_rdm_pos_energy_non_square_test
    run(system, 100.0, 50.0, 10.0, 1, [+1, -1])
