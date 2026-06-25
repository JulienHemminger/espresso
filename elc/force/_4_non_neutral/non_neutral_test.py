import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from common.generators.positions import get_rdm_constrained_points_np
from common.set_utils import are_sets_equal
from elc.force.get_custom_elc_forces import get_elc_forces
from elc.force._2_small_box_neutral.reference_method.get_ewald_forces import get_ewald_forces_2d


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

    reference_forces = get_ewald_forces_2d(system)
    elc_forces = get_elc_forces(system, gap_size, pw_error)

    # Convert forces to numpy arrays for easier plotting
    ref_f = np.array(reference_forces)
    elc_f = np.array(elc_forces)
    indices = np.arange(particle_count)

    # Plotting logic
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    components = ["X", "Y", "Z"]

    for i, ax in enumerate(axes):
        ax.scatter(
            indices,
            ref_f[:, i],
            color="black",
            marker="o",
            label="Reference (Ewald 2D)",
            s=80,
        )
        ax.scatter(
            indices,
            elc_f[:, i],
            color="crimson",
            marker="x",
            label="ELC Forces",
            s=80,
        )

        ax.set_title(f"{components[i]} Component of Force")
        ax.set_xlabel("Particle Index")
        ax.set_ylabel("Force")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()


system = espressomd.System(box_l=[10, 10, 3])
system.time_step = 0.01
system.cell_system.skin = 0.4

run_basic(system, 10, 10, 3, 1, [+0.5, -0.6, +0.9, -1.3, +0.2])