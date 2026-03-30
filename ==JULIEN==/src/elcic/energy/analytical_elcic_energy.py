import numpy as np

def analytical_elcic_energy(system, params, k_max=10, tol=1e-8):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    delta = delta_b * delta_t
    gap_size = params["gap_size"]

    print(f"--- Debugging z-positions: {positions[:, 2]} ---")

    # Pre-calculate charge products and z-distances for all pairs
    qi_qj = charges[:, np.newaxis] * charges[np.newaxis, :]
    zi, zj = positions[:, 2][:, np.newaxis], positions[:, 2][np.newaxis, :]
    dx_base = positions[:, 0][:, np.newaxis] - positions[:, 0][np.newaxis, :]
    dy_base = positions[:, 1][:, np.newaxis] - positions[:, 1][np.newaxis, :]
    
    # Pre-calculate image charge z-offsets
    k_range = np.arange(k_max + 1)
    delta_k = delta**k_range
    z_offsets = {
        "22": -(2 * k_range * lz + zj[..., np.newaxis]),
        "24": 2 * (k_range + 1) * lz - zj[..., np.newaxis],
        "23": -(2 * k_range[1:] * lz - zj[..., np.newaxis]),
        "25": 2 * k_range[1:] * lz + zj[..., np.newaxis],
    }

    total_energy = 0.0
    n = 0
    converged = False

    while not converged:
        shell_energy = 0.0
        if n == 0:
            nx_vals, ny_vals = [0], [0]
        else:
            x_ring = np.concatenate([np.full(2 * n + 1, n), np.full(2 * n + 1, -n), np.arange(-n + 1, n), np.arange(-n + 1, n)])
            y_ring = np.concatenate([np.arange(-n, n + 1), np.arange(-n, n + 1), np.full(2 * n - 2, n), np.full(2 * n - 2, -n)])
            nx_vals, ny_vals = x_ring, y_ring

        for nx, ny in zip(nx_vals, ny_vals):
            dx = dx_base - nx * lx
            dy = dy_base - ny * ly
            xy_dist_sq = dx**2 + dy**2

            # 1. Real-Real Interactions
            r_sq_real = xy_dist_sq + (zi - zj) ** 2
            if nx == 0 and ny == 0:
                np.fill_diagonal(r_sq_real, np.inf)
            
            real_contrib = 0.5 * prefactor * np.sum(qi_qj / np.sqrt(r_sq_real))
            shell_energy += real_contrib

            # 2. Image Charges
            # Eq 2.2
            dist_22 = np.sqrt(xy_dist_sq[..., np.newaxis] + (zi[..., np.newaxis] - z_offsets["22"]) ** 2)
            if np.any(dist_22 == 0): print(f"n={n}: Zero distance in Eq 2.2 at nx={nx}, ny={ny}")
            e22 = 0.5 * prefactor * np.sum((qi_qj[..., np.newaxis] * delta_k * delta_b) / dist_22)
            shell_energy += e22

            # Eq 2.4
            dist_24 = np.sqrt(xy_dist_sq[..., np.newaxis] + (zi[..., np.newaxis] - z_offsets["24"]) ** 2)
            if np.any(dist_24 == 0): print(f"n={n}: Zero distance in Eq 2.4 at nx={nx}, ny={ny}")
            e24 = 0.5 * prefactor * np.sum((qi_qj[..., np.newaxis] * delta_k * delta_t) / dist_24)
            shell_energy += e24

            # Eq 2.3
            dist_23 = np.sqrt(xy_dist_sq[..., np.newaxis] + (zi[..., np.newaxis] - z_offsets["23"]) ** 2)
            if np.any(dist_23 == 0): print(f"n={n}: Zero distance in Eq 2.3 at nx={nx}, ny={ny}")
            e23 = 0.5 * prefactor * np.sum((qi_qj[..., np.newaxis] * delta_k[1:]) / dist_23)
            shell_energy += e23

            # Eq 2.5
            dist_25 = np.sqrt(xy_dist_sq[..., np.newaxis] + (zi[..., np.newaxis] - z_offsets["25"]) ** 2)
            if np.any(dist_25 == 0): print(f"n={n}: Zero distance in Eq 2.5 at nx={nx}, ny={ny}")
            e25 = 0.5 * prefactor * np.sum((qi_qj[..., np.newaxis] * delta_k[1:]) / dist_25)
            shell_energy += e25

            if n == 0 and nx == 0 and ny == 0:
                print(f"n=0 Central Cell Contributions:")
                print(f"  Real-Real: {real_contrib}")
                print(f"  Eq 2.2 (Bot Image): {e22}")
                print(f"  Eq 2.4 (Top Image): {e24}")
                print(f"  Eq 2.3: {e23}")
                print(f"  Eq 2.5: {e25}")

        total_energy += shell_energy
        if n > 0 and abs(shell_energy) < tol:
            converged = True
        n += 1
        if n > 1000: break

    print(f"Final Total Energy: {total_energy}")
    return total_energy