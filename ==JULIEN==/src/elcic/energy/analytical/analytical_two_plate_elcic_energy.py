import numpy as np

def analytical_two_plate_elcic_energy(system, params, tol=1e-6):
    """


    * zusätzliche Summe über die Spiegelladungen in z-Richtung die bei einer Platte eben nur ein einzelner Wert waren (siehe auch mein 2D Ewald Code)


    Computes the total electrostatic energy of a 2d+h slab system with two planar dielectric interfaces by "brute force" summing a set amount of slabs (n_max) and reflections (k_max).
    Converges towards the true total electrostatic energy as n_max and k_max increase.

    n_max: the cutoff number of periodic boundary condition (PBC) "clones" of the central slab system in x and y direction
        * the total number of slabs per direction is n_max(left) + 1(center) + n_max(right)
        * in total theres (n_max + 1 + n_max)^2 slabs

    k_max: the cutoff number for reflections of image charges.
        * there are infinite reflections back and forth in z-direction between the two dielectric interfaces.
        * k_max is a cutoff, that cuts off this infinite sum to return a finite value


    Example for parameters:
        system: espressomd.System
        params = {
            "lx": 9.0,
            "ly": 12.0,
            "lz": 19.0,
            "gap_size": 15.0,
            "prefactor": 1.0,
            "delta_mid_top": -1.0,
            "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
            "charges": [+1, -1],
            'pw_error': 1e-8,
            "positions": [np.array([2, 5, 0]), np.array([8, 3, 0])],
            "title": "Dual Plates, Both Metallic, Neutral"
        }


    
    """
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    delta = delta_b * delta_t

    prev_energy = float('inf')
    i = 1
    
    while True:
        k_max = 2**i
        n_max = 10 * i
        total_energy = 0.0
        pbc_range = range(-n_max, n_max + 1)

        for idx_i in range(N):
            pos_i, qi = positions[idx_i], charges[idx_i]
            zi = pos_i[2]

            for idx_j in range(N):
                pos_j, qj = positions[idx_j], charges[idx_j]
                zj = pos_j[2]

                for nx in pbc_range:
                    for ny in pbc_range:
                        dx = pos_i[0] - (pos_j[0] + nx * lx)
                        dy = pos_i[1] - (pos_j[1] + ny * ly)
                        xy_dist_sq = dx**2 + dy**2

                        # 1. Real-Real Interactions
                        if not (idx_i == idx_j and nx == 0 and ny == 0):
                            r = np.sqrt(xy_dist_sq + (zi - zj)**2)
                            total_energy += 0.5 * prefactor * (qi * qj) / r

                        # 2. Image Charge Sequences
                        for n in range(k_max + 1):
                            # Single delta_b/t factor
                            z_img_22 = -(2 * n * lz + zj)
                            z_img_24 = 2 * (n + 1) * lz - zj
                            
                            total_energy += 0.5 * prefactor * qi * qj * (delta**n) * (
                                delta_b / np.sqrt(xy_dist_sq + (zi - z_img_22)**2) +
                                delta_t / np.sqrt(xy_dist_sq + (zi - z_img_24)**2)
                            )

                            # Pure delta^n factors
                            if n > 0:
                                z_img_23 = -(2 * n * lz - zj)
                                z_img_25 = 2 * n * lz + zj
                                total_energy += 0.5 * prefactor * qi * qj * (delta**n) * (
                                    1.0 / np.sqrt(xy_dist_sq + (zi - z_img_23)**2) +
                                    1.0 / np.sqrt(xy_dist_sq + (zi - z_img_25)**2)
                                )

        # Check convergence
        if i > 1 and abs(total_energy - prev_energy) < tol:
            return total_energy
        
        prev_energy = total_energy
        i += 1