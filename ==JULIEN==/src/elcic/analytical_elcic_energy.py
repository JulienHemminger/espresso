import numpy as np


def analytical_elcic_energy(system, params, k_max=10, tol=1e-8):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    delta = delta_b * delta_t

    # Pre-calculate charge products and z-distances for all pairs
    qi_qj = charges[:, np.newaxis] * charges[np.newaxis, :]
    zi, zj = positions[:, 2][:, np.newaxis], positions[:, 2][np.newaxis, :]
    dx_base = positions[:, 0][:, np.newaxis] - positions[:, 0][np.newaxis, :]
    dy_base = positions[:, 1][:, np.newaxis] - positions[:, 1][np.newaxis, :]

    # Pre-calculate image charge z-offsets (k_max is small, so we keep this loop)
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
        # Determine replicas for the current shell
        if n == 0:
            nx_vals, ny_vals = [0], [0]
        else:
            # Only calculate the new "ring" of replicas to avoid re-computing
            x_ring = np.concatenate(
                [
                    np.full(2 * n + 1, n),
                    np.full(2 * n + 1, -n),
                    np.arange(-n + 1, n),
                    np.arange(-n + 1, n),
                ]
            )
            y_ring = np.concatenate(
                [
                    np.arange(-n, n + 1),
                    np.arange(-n, n + 1),
                    np.full(2 * n - 2, n),
                    np.full(2 * n - 2, -n),
                ]
            )
            nx_vals, ny_vals = x_ring, y_ring

        for nx, ny in zip(nx_vals, ny_vals):
            dx = dx_base - nx * lx
            dy = dy_base - ny * ly
            xy_dist_sq = dx**2 + dy**2

            # 1. Real-Real Interactions
            r_sq_real = xy_dist_sq + (zi - zj) ** 2
            if nx == 0 and ny == 0:
                # Mask self-interaction: set diag to infinity so 1/r = 0
                np.fill_diagonal(r_sq_real, np.inf)

            shell_energy += 0.5 * prefactor * np.sum(qi_qj / np.sqrt(r_sq_real))

            # 2. Image Charges (Vectorized over k)
            # Eq 2.2 & 2.4
            shell_energy += (
                0.5
                * prefactor
                * np.sum(
                    (qi_qj[..., np.newaxis] * delta_k * delta_b)
                    / np.sqrt(
                        xy_dist_sq[..., np.newaxis]
                        + (zi[..., np.newaxis] - z_offsets["22"]) ** 2
                    )
                )
            )
            shell_energy += (
                0.5
                * prefactor
                * np.sum(
                    (qi_qj[..., np.newaxis] * delta_k * delta_t)
                    / np.sqrt(
                        xy_dist_sq[..., np.newaxis]
                        + (zi[..., np.newaxis] - z_offsets["24"]) ** 2
                    )
                )
            )
            # Eq 2.3 & 2.5
            shell_energy += (
                0.5
                * prefactor
                * np.sum(
                    (qi_qj[..., np.newaxis] * delta_k[1:])
                    / np.sqrt(
                        xy_dist_sq[..., np.newaxis]
                        + (zi[..., np.newaxis] - z_offsets["23"]) ** 2
                    )
                )
            )
            shell_energy += (
                0.5
                * prefactor
                * np.sum(
                    (qi_qj[..., np.newaxis] * delta_k[1:])
                    / np.sqrt(
                        xy_dist_sq[..., np.newaxis]
                        + (zi[..., np.newaxis] - z_offsets["25"]) ** 2
                    )
                )
            )

        total_energy += shell_energy
        if n > 0 and abs(shell_energy) < tol:
            converged = True
        n += 1

        if n > 1000:  # Safety break
            break

    return total_energy
