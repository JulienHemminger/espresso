import numpy as np
from scipy.special import erfcx, erf, erfc

def get_2d_ewald_energy(params, k_max=10, tol=1e-8):
    positions = params["positions"]
    charges = params["charges"]
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_mid_bot = params["delta_mid_bot"]
    delta_mid_top = params["delta_mid_top"]
    
    n_real = 10
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    Lx, Ly = lx, ly
    A = Lx * Ly

    alpha = 5.0 / min(Lx, Ly)

    # ------------------------------------------------------------------
    # Numerical Protection Setup
    # ------------------------------------------------------------------
    # Product of reflection coefficients
    delta_prod = delta_mid_bot * delta_mid_top
    
    # Determine necessary bounds for explicit image summation in Real Space
    # Based on: (delta_prod)^n_img * erfc(alpha * n_img * Lz) / (n_img * Lz) < tol
    if abs(delta_prod) > 1e-12:
        n_img_max = int(np.ceil(-np.log(tol) / (alpha * lz))) + 2
        n_img_max = max(5, min(n_img_max, 50)) # Keep bounded for performance
    else:
        n_img_max = 1 # Only immediate layers needed if no multi-reflection

    # ------------------------------------------------------------------
    # Helper: h(k, z) via erfcx for numerical stability
    # ------------------------------------------------------------------
    def _h(k, z):
        u_plus = k / (2 * alpha) + alpha * z
        u_minus = k / (2 * alpha) - alpha * z
        # Safeguard exponentials from overflowing
        arg_exp = -k**2 / (4 * alpha**2) - (alpha * z)**2
        if arg_exp < -700:
            return 0.0
        prefac = np.exp(arg_exp)
        return prefac * (erfcx(u_plus) + erfcx(u_minus))

    # ------------------------------------------------------------------
    # Real-space sum with Infinite Dielectric Images
    # ------------------------------------------------------------------
    E_real = 0.0
    N_atoms = len(charges)
    
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            sx = nx * Lx
            sy = ny * Ly
            
            for nz in range(-n_img_max, n_img_max + 1):
                # Calculate image magnitudes based on the geometric layer number
                # nz represents the number of round trips between the interfaces
                if nz == 0:
                    q_fac_even = 1.0
                    q_fac_odd_bot = delta_mid_bot
                    q_fac_odd_top = delta_mid_top
                else:
                    q_fac_even = delta_prod ** abs(nz)
                    q_fac_odd_bot = delta_mid_bot * (delta_prod ** abs(nz))
                    q_fac_odd_top = delta_mid_top * (delta_prod ** abs(nz))
                
                for i in range(N_atoms):
                    for j in range(N_atoms):
                        dx_base = positions[j, 0] - positions[i, 0] + sx
                        dy_base = positions[j, 1] - positions[i, 1] + sy
                        
                        # --- 1. Even-reflection/Source Image series ---
                        # z_image = z_j + 2 * nz * Lz
                        if not (nx == 0 and ny == 0 and nz == 0 and i == j):
                            dz_even = (positions[j, 2] + 2.0 * nz * lz) - positions[i, 2]
                            r_even = np.sqrt(dx_base**2 + dy_base**2 + dz_even**2)
                            if r_even > 1e-14:
                                E_real += charges[i] * charges[j] * q_fac_even * erfc(alpha * r_even) / r_even
                        
                        # --- 2. Odd-reflection (Bottom Interface) series ---
                        # z_image = -z_j + 2 * nz * Lz
                        dz_odd_bot = (-positions[j, 2] + 2.0 * nz * lz) - positions[i, 2]
                        r_odd_bot = np.sqrt(dx_base**2 + dy_base**2 + dz_odd_bot**2)
                        if r_odd_bot > 1e-14:
                            E_real += charges[i] * charges[j] * q_fac_odd_bot * erfc(alpha * r_odd_bot) / r_odd_bot
                            
                        # --- 3. Odd-reflection (Top Interface) series ---
                        # z_image = 2 * Lz - z_j + 2 * nz * Lz
                        dz_odd_top = (2.0 * lz - positions[j, 2] + 2.0 * nz * lz) - positions[i, 2]
                        r_odd_top = np.sqrt(dx_base**2 + dy_base**2 + dz_odd_top**2)
                        if r_odd_top > 1e-14:
                            E_real += charges[i] * charges[j] * q_fac_odd_top * erfc(alpha * r_odd_top) / r_odd_top

    E_real *= 0.5

    # ------------------------------------------------------------------
    # Reciprocal-space sum (k != 0) using analytical image sum
    # ------------------------------------------------------------------
    E_recip = 0.0
    two_pi_over_Lx = 2 * np.pi / Lx
    two_pi_over_Ly = 2 * np.pi / Ly
    
    for mx in range(-k_max, k_max + 1):
        for my in range(-k_max, k_max + 1):
            if mx == 0 and my == 0:
                continue
            
            kx = two_pi_over_Lx * mx
            ky = two_pi_over_Ly * my
            k = np.sqrt(kx * kx + ky * ky)
            
            # Reciprocal space structure factors including the geometric sum of images
            # Denominator for parallel dielectric slab summation: 1 - delta_prod * e^(-2*k*Lz)
            denom = 1.0 - delta_prod * np.exp(-2.0 * k * lz)
            if abs(denom) < 1e-14:
                denom = 1e-14 # Safe guard division by zero
                
            for i in range(N_atoms):
                for j in range(N_atoms):
                    drho_x = positions[j, 0] - positions[i, 0]
                    drho_y = positions[j, 1] - positions[i, 1]
                    k_dot_rho = kx * drho_x + ky * drho_y
                    cos_fac = np.cos(k_dot_rho)
                    
                    zi = positions[i, 2]
                    zj = positions[j, 2]
                    
                    # Core interaction types inside the dielectric cavity
                    term_source = _h(k, zj - zi) + delta_prod * np.exp(-2.0 * k * lz) * _h(k, -(zj - zi))
                    term_bot = delta_mid_bot * (_h(k, zi + zj) + delta_mid_top * np.exp(-2.0 * k * lz) * _h(k, -(zi + zj)))
                    term_top = delta_mid_top * np.exp(-2.0 * k * lz) * (_h(k, 2.0 * lz - (zi + zj)) + delta_mid_bot * _h(k, -(2.0 * lz - (zi + zj))))
                    
                    E_ij = (term_source + term_bot + term_top) / denom
                    E_recip += charges[i] * charges[j] * cos_fac * E_ij / k

    E_recip *= 0.5 * (np.pi / A)

    # ------------------------------------------------------------------
    # k = 0 Analytical Term
    # ------------------------------------------------------------------
    def _g(z):
        abs_z = np.abs(z)
        return abs_z * erf(alpha * abs_z) + np.exp(-(alpha * z)**2) / (alpha * np.sqrt(np.pi))

    E_k0 = 0.0
    # Process k=0 explicitly to protect against uniform background / system charge neutralizing issues
    # Under two interfaces, the 1D planar potential must sum all images up to a threshold convergence
    k0_img_limit = max(40, n_img_max * 2)
    
    for i in range(N_atoms):
        for j in range(N_atoms):
            q_ij = charges[i] * charges[j]
            zi = positions[i, 2]
            zj = positions[j, 2]
            
            s_k0 = 0.0
            for nz in range(-k0_img_limit, k0_img_limit + 1):
                if nz == 0:
                    q_fac_even = 1.0
                    q_fac_odd_bot = delta_mid_bot
                    q_fac_odd_top = delta_mid_top
                else:
                    q_fac_even = delta_prod ** abs(nz)
                    q_fac_odd_bot = delta_mid_bot * (delta_prod ** abs(nz))
                    q_fac_odd_top = delta_mid_top * (delta_prod ** abs(nz))
                
                s_k0 += q_fac_even * _g(zj + 2.0 * nz * lz - zi)
                s_k0 += q_fac_odd_bot * _g(-zj + 2.0 * nz * lz - zi)
                s_k0 += q_fac_odd_top * _g(2.0 * lz - zj + 2.0 * nz * lz - zi)
                
            E_k0 += q_ij * s_k0

    E_k0 *= -1.0 * (np.pi / A)

    # ------------------------------------------------------------------
    # Self-energy Correction
    # ------------------------------------------------------------------
    E_self = -(alpha / np.sqrt(np.pi)) * np.sum(charges**2)

    # Sum up totals
    E_total = E_real + E_recip + E_k0 + E_self
    return prefactor * E_total