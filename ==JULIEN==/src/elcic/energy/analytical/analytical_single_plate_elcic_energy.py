import numpy as np
from scipy.special import erfcx, erf, erfc

def analytical_single_plate_2d_ewald_elcic_energy(positions, charges, box_l, prefactor, delta_mid_bot, 
                                                 accuracy=1e-8, alpha=None, max_iter=50):
    """
    Computes 2D Ewald energy with adaptive convergence for k_max and n_real.
    """
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)
    Lx, Ly = box_l[0], box_l[1]
    A = Lx * Ly

    if alpha is None:
        alpha = 5.0 / min(Lx, Ly)

    # Mirror charges
    pos_mirror = positions.copy()
    pos_mirror[:, 2] = -pos_mirror[:, 2]
    q_mirror = delta_mid_bot * charges

    def _h(k, z):
        u_plus = k / (2 * alpha) + alpha * z
        u_minus = k / (2 * alpha) - alpha * z
        # Using erfcx(x) = exp(x^2)erfc(x) to avoid overflow
        return np.exp(-k**2 / (4 * alpha**2) - (alpha * z)**2) * (erfcx(u_plus) + erfcx(u_minus))

    # --- Constant Terms (Self and K=0) ---
    def get_constant_terms():
        # Self energy
        E_self = -(alpha / np.sqrt(np.pi)) * np.sum(charges**2)
        
        # K=0 term for both real-real and real-mirror
        def k0_calc(pos_a, q_a, pos_b, q_b):
            # Vectorized z differences
            z_ij = pos_b[:, 2][None, :] - pos_a[:, 2][:, None]
            abs_z = np.abs(z_ij)
            term = abs_z * erf(alpha * abs_z) + np.exp(-(alpha * z_ij)**2) / (alpha * np.sqrt(np.pi))
            return - (np.pi / A) * np.sum(q_a[:, None] * q_b[None, :] * term)

        return E_self + k0_calc(positions, charges, positions, charges) + k0_calc(positions, charges, pos_mirror, q_mirror)

    # --- Real Space Shell ---
    def get_real_shell(n):
        """Calculates contribution of the shell at index n (square perimeter)"""
        E_shell = 0.0
        # Determine indices for the square shell at distance n
        indices = []
        for i in range(-n, n + 1):
            indices.append((i, n))
            indices.append((i, -n))
        for j in range(-(n - 1), n):
            indices.append((n, j))
            indices.append((-n, j))

        for nx, ny in indices:
            sx, sy = nx * Lx, ny * Ly
            
            # Sub-function for interaction between two sets
            def pair_real(pos_a, q_a, pos_b, q_b, exclude_self):
                dx = pos_b[:, 0][None, :] - pos_a[:, 0][:, None] + sx
                dy = pos_b[:, 1][None, :] - pos_a[:, 1][:, None] + sy
                dz = pos_b[:, 2][None, :] - pos_a[:, 2][:, None]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Mask for self-interaction at n=(0,0)
                if exclude_self and nx == 0 and ny == 0:
                    diag_mask = ~np.eye(len(q_a), dtype=bool)
                    return np.sum((q_a[:, None] * q_b[None, :] * erfc(alpha * r) / r)[diag_mask])
                return np.sum(q_a[:, None] * q_b[None, :] * erfc(alpha * r) / r)

            E_shell += pair_real(positions, charges, positions, charges, True)
            E_shell += pair_real(positions, charges, pos_mirror, q_mirror, False)
        
        return 0.5 * E_shell

    # --- Reciprocal Space Shell ---
    def get_recip_shell(k_idx):
        """Calculates contribution of the shell at index k_idx"""
        E_shell = 0.0
        indices = []
        for i in range(-k_idx, k_idx + 1):
            indices.append((i, k_idx))
            indices.append((i, -k_idx))
        for j in range(-(k_idx - 1), k_idx):
            indices.append((k_idx, j))
            indices.append((-k_idx, j))

        two_pi_L = 2 * np.pi / box_l[:2]
        
        for mx, my in indices:
            if mx == 0 and my == 0: continue
            kx, ky = mx * two_pi_L[0], my * two_pi_L[1]
            k = np.sqrt(kx**2 + ky**2)
            
            def pair_recip(pos_a, q_a, pos_b, q_b):
                drho_x = pos_b[:, 0][None, :] - pos_a[:, 0][:, None]
                drho_y = pos_b[:, 1][None, :] - pos_a[:, 1][:, None]
                z_ij = pos_b[:, 2][None, :] - pos_a[:, 2][:, None]
                k_dot_rho = kx * drho_x + ky * drho_y
                return np.sum(q_a[:, None] * q_b[None, :] * np.cos(k_dot_rho) * _h(k, z_ij) / k)

            E_shell += pair_recip(positions, charges, positions, charges)
            E_shell += pair_recip(positions, charges, pos_mirror, q_mirror)
            
        return 0.5 * (np.pi / A) * E_shell

    # --- Convergence Loop ---
    total_energy = get_constant_terms()
    
    # Iterate shells until both contributions fall below accuracy
    for n in range(0, max_iter):
        delta_real = get_real_shell(n)
        delta_recip = get_recip_shell(n) if n > 0 else 0.0
        
        total_energy += delta_real + delta_recip
        
        # Check convergence after at least one shell
        if n > 0 and abs(delta_real) < accuracy and abs(delta_recip) < accuracy:
            break
            
    return prefactor * total_energy