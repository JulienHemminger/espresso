import numpy as np


def analytical_elcic_energy(system, params, k_max=10, n_max=2**8):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    delta = delta_b * delta_t

    total_energy = 0.0

    # Define the range of periodic replicas in x and y
    pbc_range = range(-n_max, n_max + 1)

    for i in range(N):
        pos_i = positions[i]
        qi = charges[i]

        for j in range(N):
            pos_j = positions[j]
            qj = charges[j]

            for nx in pbc_range:
                for ny in pbc_range:
                    # Shift j-th particle by periodic box vectors
                    dx = pos_i[0] - (pos_j[0] + nx * lx)
                    dy = pos_i[1] - (pos_j[1] + ny * ly)
                    xy_dist_sq = dx**2 + dy**2

                    # 1. Real-Real Interactions (Exclude self-interaction in central cell)
                    if not (i == j and nx == 0 and ny == 0):
                        r = np.sqrt(xy_dist_sq + (pos_i[2] - pos_j[2]) ** 2)
                        total_energy += 0.5 * prefactor * (qi * qj) / r

                    # 2. Image Charge Sequences (All replicas)
                    zi, zj = pos_i[2], pos_j[2]
                    for n in range(k_max + 1):
                        # Eq 2.2 & 2.4 (Single delta_b/t factor)
                        z_img_22 = -(2 * n * lz + zj)
                        z_img_24 = 2 * (n + 1) * lz - zj

                        total_energy += (
                            0.5
                            * prefactor
                            * (qi * qj * (delta**n) * delta_b)
                            / np.sqrt(xy_dist_sq + (zi - z_img_22) ** 2)
                        )
                        total_energy += (
                            0.5
                            * prefactor
                            * (qi * qj * (delta**n) * delta_t)
                            / np.sqrt(xy_dist_sq + (zi - z_img_24) ** 2)
                        )

                        # Eq 2.3 & 2.5 (Pure delta^n factors)
                        if n > 0:
                            z_img_23 = -(2 * n * lz - zj)
                            z_img_25 = 2 * n * lz + zj
                            total_energy += (
                                0.5
                                * prefactor
                                * (qi * qj * (delta**n))
                                / np.sqrt(xy_dist_sq + (zi - z_img_23) ** 2)
                            )
                            total_energy += (
                                0.5
                                * prefactor
                                * (qi * qj * (delta**n))
                                / np.sqrt(xy_dist_sq + (zi - z_img_25) ** 2)
                            )

    return total_energy
