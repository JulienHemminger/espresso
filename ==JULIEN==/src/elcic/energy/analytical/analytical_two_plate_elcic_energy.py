import numpy as np

def analytical_two_plate_elcic_energy(system, params, k_max=10, tol=1e-8):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    delta = delta_b * delta_t
    
    qi_qj = charges[:, np.newaxis] * charges[np.newaxis, :]
    zi, zj = positions[:, 2][:, np.newaxis], positions[:, 2][np.newaxis, :]
    dx_base = positions[:, 0][:, np.newaxis] - positions[:, 0][np.newaxis, :]
    dy_base = positions[:, 1][:, np.newaxis] - positions[:, 1][np.newaxis, :]

    # Pre-calculate Image Charge factors
    k_range = np.arange(k_max + 1)
    delta_k = delta**k_range
    
    # Pre-calculate z_offsets as flat arrays for broadcasting
    z_offs = {
        "22": -(2 * k_range * lz + zj[..., np.newaxis]),
        "24": 2 * (k_range + 1) * lz - zj[..., np.newaxis],
        "23": -(2 * k_range[1:] * lz - zj[..., np.newaxis]),
        "25": 2 * k_range[1:] * lz + zj[..., np.newaxis],
    }

    total_energy = 0.0
    n = 0
    max_shells = 100 

    while n < max_shells:
        # Vectorize the current shell's nx, ny pairs
        if n == 0:
            nx_vals, ny_vals = np.array([0]), np.array([0])
        else:
            # Create a full grid for the shell for faster batch processing
            r = np.arange(-n, n + 1)
            nx, ny = np.meshgrid(r, r)
            mask = (np.abs(nx) == n) | (np.abs(ny) == n)
            nx_vals, ny_vals = nx[mask], ny[mask]

        # Broadcast nx/ny to (N, N, Shell_Size)
        dx = dx_base[..., np.newaxis] - nx_vals * lx
        dy = dy_base[..., np.newaxis] - ny_vals * ly
        xy_dist_sq = dx**2 + dy**2

        # 1. Real-Real
        r_sq_real = xy_dist_sq + (zi[..., np.newaxis] - zj[..., np.newaxis])**2
        if n == 0:
            np.fill_diagonal(r_sq_real[:, :, 0], np.inf)
        
        shell_energy = 0.5 * prefactor * np.sum(qi_qj[..., np.newaxis] / np.sqrt(r_sq_real))

        # 2. Image Charges
        # We process all nx, ny in the shell simultaneously
        for key, weight, dk in [("22", delta_b, delta_k), ("24", delta_t, delta_k), 
                                ("23", 1.0, delta_k[1:]), ("25", 1.0, delta_k[1:])]:
            # Shape: (N, N, Shell_Size, K_range)
            z_diff_sq = (zi[..., np.newaxis, np.newaxis] - z_offs[key][..., np.newaxis, :])**2
            inv_r = 1.0 / np.sqrt(xy_dist_sq[..., np.newaxis] + z_diff_sq)
            shell_energy += 0.5 * prefactor * np.sum(qi_qj[..., np.newaxis, np.newaxis] * dk * weight * inv_r)

        total_energy += shell_energy
        if n > 0 and abs(shell_energy) < tol:
            break
        n += 1

    return total_energy