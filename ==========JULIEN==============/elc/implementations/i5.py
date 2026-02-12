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
    
    # ELC correction calculation
    ux = 1.0 / lx
    uy = 1.0 / ly
    uz = 1.0 / lz
    
    # Height of extended periodic box (Lz in the paper)
    Lz = lz + gap_size
    
    # Calculate L1 for dipole correction (Eq. 3.3)
    L1 = np.sum(qs * zs)
    
    # Determine cutoff for p, q based on exponential decay
    # We want exp(-2π·f_pq·gap_size) < pw_error
    f_cutoff = -np.log(pw_error) / (2 * np.pi * gap_size)
    p_max = int(np.ceil(f_cutoff / ux)) + 1
    q_max = int(np.ceil(f_cutoff / uy)) + 1
    
    elc_energy = 0.0
    
    # Sum over (p,q) where p² + q² > 0
    for p in range(-p_max, p_max + 1):
        for q in range(-q_max, q_max + 1):
            if p == 0 and q == 0:
                continue
            
            # Calculate f_pq (Eq. 3.2)
            f_pq = np.sqrt((ux * p)**2 + (uy * q)**2)
            
            # Calculate ρ_p and σ_q (Eq. 3.2)
            rho_p = 2 * np.pi * ux * p
            sigma_q = 2 * np.pi * uy * q
            
            # Calculate Λ factors for lower replicas (Eq. 3.3)
            # These use exp(-2π·f_pq·zi)
            exp_minus = np.exp(-2 * np.pi * f_pq * zs)
            lambda_minus_cc = np.sum(qs * exp_minus * np.cos(rho_p * xs) * np.cos(sigma_q * ys))
            lambda_minus_sc = np.sum(qs * exp_minus * np.sin(rho_p * xs) * np.cos(sigma_q * ys))
            lambda_minus_cs = np.sum(qs * exp_minus * np.cos(rho_p * xs) * np.sin(sigma_q * ys))
            lambda_minus_ss = np.sum(qs * exp_minus * np.sin(rho_p * xs) * np.sin(sigma_q * ys))
            
            # Calculate Λ factors for upper replicas (Eq. 3.3)
            # These use exp(+2π·f_pq·zi)
            exp_plus = np.exp(2 * np.pi * f_pq * zs)
            lambda_plus_cc = np.sum(qs * exp_plus * np.cos(rho_p * xs) * np.cos(sigma_q * ys))
            lambda_plus_sc = np.sum(qs * exp_plus * np.sin(rho_p * xs) * np.cos(sigma_q * ys))
            lambda_plus_cs = np.sum(qs * exp_plus * np.cos(rho_p * xs) * np.sin(sigma_q * ys))
            lambda_plus_ss = np.sum(qs * exp_plus * np.sin(rho_p * xs) * np.sin(sigma_q * ys))
            
            # Calculate ℒ_pq function (Eq. 3.7)
            exp_4Lz = np.exp(-4 * np.pi * Lz * f_pq)
            if exp_4Lz >= 1.0:  # Skip if denominator would be problematic
                continue
            
            # X factors for lower replicas (Eq. 3.6)
            # X^(L−,+) uses ℒ_pq(zi)
            L_pq_z = np.exp(-2 * np.pi * f_pq * zs) / (1 - exp_4Lz)
            X_lower_cc = np.sum(qs * L_pq_z * np.cos(rho_p * xs) * np.cos(sigma_q * ys))
            X_lower_sc = np.sum(qs * L_pq_z * np.sin(rho_p * xs) * np.cos(sigma_q * ys))
            X_lower_cs = np.sum(qs * L_pq_z * np.cos(rho_p * xs) * np.sin(sigma_q * ys))
            X_lower_ss = np.sum(qs * L_pq_z * np.sin(rho_p * xs) * np.sin(sigma_q * ys))
            
            # X factors for upper replicas (Eq. 3.6)
            # X^(L+,−) uses ℒ_pq(Lz - zi)
            L_pq_Lz_minus_z = np.exp(-2 * np.pi * f_pq * (Lz - zs)) / (1 - exp_4Lz)
            X_upper_cc = np.sum(qs * L_pq_Lz_minus_z * np.cos(rho_p * xs) * np.cos(sigma_q * ys))
            X_upper_sc = np.sum(qs * L_pq_Lz_minus_z * np.sin(rho_p * xs) * np.cos(sigma_q * ys))
            X_upper_cs = np.sum(qs * L_pq_Lz_minus_z * np.cos(rho_p * xs) * np.sin(sigma_q * ys))
            X_upper_ss = np.sum(qs * L_pq_Lz_minus_z * np.sin(rho_p * xs) * np.sin(sigma_q * ys))
            
            # First term in Eq. 3.5: Λ^(−) · X^(L−,+)
            term1 = (lambda_minus_cc * X_lower_cc + lambda_minus_sc * X_lower_sc + 
                     lambda_minus_cs * X_lower_cs + lambda_minus_ss * X_lower_ss)
            
            # Second term in Eq. 3.5: X^(L+,−) · Λ^(+)
            term2 = (X_upper_cc * lambda_plus_cc + X_upper_sc * lambda_plus_sc + 
                     X_upper_cs * lambda_plus_cs + X_upper_ss * lambda_plus_ss)
            
            # Add this (p,q) contribution to ELC energy
            elc_energy += -0.5 * ux * uy / f_pq * (term1 + term2)
    
    # Add dipole correction term (third term in Eq. 3.5)
    elc_energy += 2 * ux * uy * uz * L1**2
    
    # Add r_perp² term (fourth term in Eq. 3.5)
    r_perp_sq = xs**2 + ys**2
    elc_energy += -2.0/3.0 * ux * uy * uz * np.sum(qs * r_perp_sq)
    
    # E_2D+h = E_3D + ΔE_ELC
    return e_3d + elc_energy