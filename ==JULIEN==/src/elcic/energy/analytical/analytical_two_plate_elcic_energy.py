import numpy as np

def analytical_two_plate_elcic_energy(system, params, k_max=10, n_max=50):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    
    total_energy = 0.0
    pbc_range = range(-n_max, n_max + 1)

    for i in range(N):
        pos_i = positions[i]
        qi = charges[i]
        for j in range(N):
            pos_j = positions[j]
            qj = charges[j]
            zi, zj = pos_i[2], pos_j[2]

            for nx in pbc_range:
                for ny in pbc_range:
                    dx = pos_i[0] - (pos_j[0] + nx * lx)
                    dy = pos_i[1] - (pos_j[1] + ny * ly)
                    r_xy_sq = dx**2 + dy**2

                    # 1. Direct Interaction (Real charges)
                    if not (i == j and nx == 0 and ny == 0):
                        r = np.sqrt(r_xy_sq + (zi - zj)**2)
                        total_energy += 0.5 * prefactor * (qi * qj) / r

                    # 2. Image Charge Series
                    # k represents the number of round-trips/reflections
                    for k in range(k_max + 1):
                        # Factor common to reflections: (delta_b * delta_t)^k
                        amp_k = (delta_b * delta_t)**k
                        
                        # Type A: Image of qj reflected across bottom then k pairs
                        # Position: z = -2*k*Lz - zj
                        if not (k == 0 and delta_b == 0):
                            z_a = -2 * k * lz - zj
                            total_energy += 0.5 * prefactor * (qi * qj * amp_k * delta_b) / np.sqrt(r_xy_sq + (zi - z_a)**2)

                        # Type B: Image of qj reflected across top then k pairs
                        # Position: z = 2*(k+1)*Lz - zj
                        if not (k == 0 and delta_t == 0):
                            z_b = 2 * (k + 1) * lz - zj
                            total_energy += 0.5 * prefactor * (qi * qj * amp_k * delta_t) / np.sqrt(r_xy_sq + (zi - z_b)**2)

                        # Type C & D: Higher order reflections (k > 0)
                        if k > 0:
                            # Position: z = 2*k*Lz + zj
                            z_c = 2 * k * lz + zj
                            total_energy += 0.5 * prefactor * (qi * qj * amp_k) / np.sqrt(r_xy_sq + (zi - z_c)**2)
                            
                            # Position: z = -2*k*Lz + zj
                            z_d = -2 * k * lz + zj
                            total_energy += 0.5 * prefactor * (qi * qj * amp_k) / np.sqrt(r_xy_sq + (zi - z_d)**2)

    return total_energy