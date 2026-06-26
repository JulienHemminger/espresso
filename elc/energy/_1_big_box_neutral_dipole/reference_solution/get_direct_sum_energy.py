import numpy as np


def get_direct_sum_energy(system, prefactor=1.0):
    """
    Calculates the electrostatic potential energy using scalar math
    and nested loops for clarity.
    """
    pos = system.part.all().pos
    q = system.part.all().q
    n_particles = len(q)

    total_energy = 0.0

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            p1 = pos[i]
            p2 = pos[j]

            # Calculate scalar distance: sqrt(dx^2 + dy^2 + dz^2)
            r = np.linalg.norm(p1 - p2)

            # Add interaction energy for this pair
            total_energy += (q[i] * q[j]) / r

    return prefactor * total_energy
