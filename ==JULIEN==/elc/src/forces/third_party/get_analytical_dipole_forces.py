import numpy as np


def get_analytical_forces(system):
    # 1. Initialize a list of zero vectors for each particle
    particles = list(system.part.all())
    n = len(particles)
    # Using a dictionary or list to store forces mapped to indices
    forces = [np.zeros(3) for _ in range(n)]

    # ke is the Coulomb constant; adjust based on your simulation units
    ke = 1.0

    # 2. Double loop for pair-wise interactions
    for i in range(n):
        for j in range(i + 1, n):
            p1 = particles[i]
            p2 = particles[j]

            # Distance vector and magnitude
            r_vec = p1.pos - p2.pos
            dist_sq = np.sum(r_vec**2)
            dist = np.sqrt(dist_sq)

            if dist == 0:
                continue  # Avoid division by zero for overlapping particles

            # Coulomb's Law calculation
            # F = ke * (q1 * q2 / r^2) * (r_vec / r)
            force_mag = ke * (p1.q * p2.q) / dist_sq
            force_vec = force_mag * (r_vec / dist)

            # 3. Accumulate forces (Action = -Reaction)
            forces[i] += force_vec
            forces[j] -= force_vec

    return forces