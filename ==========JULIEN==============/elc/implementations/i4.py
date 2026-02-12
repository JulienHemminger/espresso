import numpy as np

def get_elc_energy(p3m, gap_size, pw_error, system):
    # https://claude.ai/chat/404cb539-dcf4-4e77-b8e6-72d975d10223
    lx, ly, lz = system.box_l
    
    parts = system.part.all()
    qs = parts.q
    pos = parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    
    # Calculate reciprocal space parameters
    ux = 1.0 / lx
    uy = 1.0 / ly
    uz = 1.0 / lz
    
    # Determine cutoff for f_pq based on convergence criterion
    # We want exp(-2π f_pq δ) < pw_error
    f_cutoff = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    
    # Determine maximum p and q to search
    max_p = int(np.ceil(f_cutoff / ux)) + 1
    max_q = int(np.ceil(f_cutoff / uy)) + 1
    
    # Calculate L₀ and L₁ (Eq. 3.3)
    L0 = np.sum(qs)
    L1 = np.sum(qs * zs)
    
    # Calculate r_perp² term
    r_perp_sq_sum = np.sum(qs * (xs**2 + ys**2))
    
    # Initialize ELC correction energy
    correction = 0.0
    
    # Sum over p, q (excluding p=q=0)
    for p in range(-max_p, max_p + 1):
        for q in range(-max_q, max_q + 1):
            if p == 0 and q == 0:
                continue
            
            # Calculate f_pq (Eq. 3.2)
            f_pq = np.sqrt((ux * p)**2 + (uy * q)**2)
            
            # Skip if outside convergence radius
            if f_pq > f_cutoff:
                continue
            
            # Calculate angular frequencies (Eq. 3.2)
            rho_p = 2.0 * np.pi * ux * p
            sigma_q = 2.0 * np.pi * uy * q
            
            # Precompute trigonometric factors
            cos_rho_x = np.cos(rho_p * xs)
            sin_rho_x = np.sin(rho_p * xs)
            cos_sigma_y = np.cos(sigma_q * ys)
            sin_sigma_y = np.sin(sigma_q * ys)
            
            # Calculate Λ^(L−,...) factors with exp(-2π f_pq z) (Eq. 3.3)
            exp_neg = np.exp(-2.0 * np.pi * f_pq * zs)
            Lambda_neg_cc = np.sum(qs * exp_neg * cos_rho_x * cos_sigma_y)
            Lambda_neg_sc = np.sum(qs * exp_neg * sin_rho_x * cos_sigma_y)
            Lambda_neg_cs = np.sum(qs * exp_neg * cos_rho_x * sin_sigma_y)
            Lambda_neg_ss = np.sum(qs * exp_neg * sin_rho_x * sin_sigma_y)
            
            # Calculate Λ^(L+,...) factors with exp(+2π f_pq z) (Eq. 3.3)
            exp_pos = np.exp(2.0 * np.pi * f_pq * zs)
            Lambda_pos_cc = np.sum(qs * exp_pos * cos_rho_x * cos_sigma_y)
            Lambda_pos_sc = np.sum(qs * exp_pos * sin_rho_x * cos_sigma_y)
            Lambda_pos_cs = np.sum(qs * exp_pos * cos_rho_x * sin_sigma_y)
            Lambda_pos_ss = np.sum(qs * exp_pos * sin_rho_x * sin_sigma_y)
            
            # Calculate ℒ_pq(z) for X^(L−,+,...) (Eq. 3.6, 3.7)
            denom = 1.0 - np.exp(-4.0 * np.pi * lz * f_pq)
            L_pq_z = np.exp(-2.0 * np.pi * f_pq * zs) / denom
            X_lower_cc = np.sum(qs * L_pq_z * cos_rho_x * cos_sigma_y)
            X_lower_sc = np.sum(qs * L_pq_z * sin_rho_x * cos_sigma_y)
            X_lower_cs = np.sum(qs * L_pq_z * cos_rho_x * sin_sigma_y)
            X_lower_ss = np.sum(qs * L_pq_z * sin_rho_x * sin_sigma_y)
            
            # Calculate ℒ_pq(Lz - z) for X^(L+,−,...) (Eq. 3.6, 3.7)
            L_pq_Lz_z = np.exp(-2.0 * np.pi * f_pq * (lz - zs)) / denom
            X_upper_cc = np.sum(qs * L_pq_Lz_z * cos_rho_x * cos_sigma_y)
            X_upper_sc = np.sum(qs * L_pq_Lz_z * sin_rho_x * cos_sigma_y)
            X_upper_cs = np.sum(qs * L_pq_Lz_z * cos_rho_x * sin_sigma_y)
            X_upper_ss = np.sum(qs * L_pq_Lz_z * sin_rho_x * sin_sigma_y)
            
            # First summation term: Λ^(L−) * X^(L−,+) (Eq. 3.5, line 1)
            term1 = (Lambda_neg_cc * X_lower_cc + 
                    Lambda_neg_sc * X_lower_sc +
                    Lambda_neg_cs * X_lower_cs +
                    Lambda_neg_ss * X_lower_ss)
            
            # Second summation term: X^(L+,−) * Λ^(L+) (Eq. 3.5, line 2)
            term2 = (X_upper_cc * Lambda_pos_cc +
                    X_upper_sc * Lambda_pos_sc +
                    X_upper_cs * Lambda_pos_cs +
                    X_upper_ss * Lambda_pos_ss)
            
            # Add to correction with -½ ux uy / f_pq prefactor
            correction -= 0.5 * ux * uy * (term1 + term2) / f_pq
    
    # Add dipole term: +2 ux uy uz L₁² (Eq. 3.5, line 3)
    correction += 2.0 * ux * uy * uz * L1**2
    
    # Add r_perp² term: -⅔ ux uy uz Σ qi ri⊥² (Eq. 3.5, line 4)
    correction -= (2.0/3.0) * ux * uy * uz * r_perp_sq_sum
    
    # Return E_2D+h = E_3D + correction (Eq. 3.5)
    return e_3d + correction