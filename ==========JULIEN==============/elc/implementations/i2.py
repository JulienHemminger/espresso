import numpy as np

def get_elc_energy(p3m, gap_size, pw_error, system):
    """
    Calculate electrostatic energy of a neutral 2D+h system using ELC.
    
    Parameters:
    -----------
    p3m : P3M solver object
    gap_size : float
        The physical gap Delta introduced to decouple replicas.
    pw_error : float
        Target pairwise error.
    system : ESPResSo system object
    """
    # 1. Physical Box dimensions
    lx, ly, lz = system.box_l
    ux, uy, uz = 1.0/lx, 1.0/ly, 1.0/lz
    
    # Get particle data
    parts = system.part.all()
    qs = parts.q
    pos = parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # 2. Calculate E_3D using the underlying solver [cite: 365, 366]
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    
    # 3. Summation cutoff R based on the gap_size [cite: 340, 342]
    # The exponential decay depends on the gap height.
    R_max = int(np.ceil(np.sqrt(-np.log(pw_error) / (2.0 * np.pi * gap_size * ux))))
    R_max = max(R_max, 10)
    
    # 4. Prepare constants for Dipole and Vector Dipole terms 
    xi_L_1 = np.sum(qs * zs) # Total dipole moment in z [cite: 368]
    # Vector dipole moment components 
    mx = np.sum(qs * xs)
    my = np.sum(qs * ys)
    mz = xi_L_1
    M_sq = mx**2 + my**2 + mz**2 # Square of total dipole vector 
    
    elc_corr = 0.0
    
    # 5. Sum over (p,q) pairs [cite: 366, 368]
    for p in range(-R_max, R_max + 1):
        for q in range(-R_max, R_max + 1):
            if p == 0 and q == 0: continue
            if p*p + q*q > R_max*R_max: continue
            
            f_pq = np.sqrt((ux * p)**2 + (uy * q)**2) # [cite: 337]
            wp = 2.0 * np.pi * ux * p # [cite: 338]
            wq = 2.0 * np.pi * uy * q # [cite: 338]
            
            # Fourier and Exponential factors for current (p,q) [cite: 346]
            cos_x, sin_x = np.cos(wp * xs), np.sin(wp * xs)
            cos_y, sin_y = np.cos(wq * ys), np.sin(wq * ys)
            exp_p = np.exp(2.0 * np.pi * f_pq * zs)
            exp_m = np.exp(-2.0 * np.pi * f_pq * zs)
            
            # Structure factors chi [cite: 346, 370]
            chi_p_cc = np.sum(qs * exp_p * cos_x * cos_y)
            chi_p_sc = np.sum(qs * exp_p * sin_x * cos_y)
            chi_p_cs = np.sum(qs * exp_p * cos_x * sin_y)
            chi_p_ss = np.sum(qs * exp_p * sin_x * sin_y)
            
            chi_m_cc = np.sum(qs * exp_m * cos_x * cos_y)
            chi_m_sc = np.sum(qs * exp_m * sin_x * cos_y)
            chi_m_cs = np.sum(qs * exp_m * cos_x * sin_y)
            chi_m_ss = np.sum(qs * exp_m * sin_x * sin_y)
            
            # Replica analytic factors L' using box height lz 
            # Note: denominator uses the periodic replication length lz.
            denom = 1.0 - np.exp(-4.0 * np.pi * lz * f_pq)
            L_prime_z = exp_m / denom
            L_prime_lz_z = np.exp(-2.0 * np.pi * f_pq * (lz - zs)) / denom
            
            # Replica factors X [cite: 372, 376]
            X_m_cc = np.sum(qs * L_prime_z * cos_x * cos_y)
            X_m_sc = np.sum(qs * L_prime_z * sin_x * cos_y)
            X_m_cs = np.sum(qs * L_prime_z * cos_x * sin_y)
            X_m_ss = np.sum(qs * L_prime_z * sin_x * sin_y)
            
            X_p_cc = np.sum(qs * L_prime_lz_z * cos_x * cos_y)
            X_p_sc = np.sum(qs * L_prime_lz_z * sin_x * cos_y)
            X_p_cs = np.sum(qs * L_prime_lz_z * cos_x * sin_y)
            X_p_ss = np.sum(qs * L_prime_lz_z * sin_x * sin_y)
            
            # First and Second summation lines from Eq 3.5 [cite: 366, 368]
            term1 = (chi_m_cc * X_p_cc + chi_m_sc * X_p_sc + 
                     chi_m_cs * X_p_cs + chi_m_ss * X_p_ss)
            term2 = (X_m_cc * chi_p_cc + X_m_sc * chi_p_sc + 
                     X_m_cs * chi_p_cs + X_m_ss * chi_p_ss)
            
            elc_corr -= 0.5 * ux * uy * (term1 + term2) / f_pq

    # 6. Dipole Correction Terms 
    # Term 1: 2*pi * ux * uy * uz * (xi_L_1)^2
    dipole_z = 2.0 * np.pi * ux * uy * uz * (xi_L_1**2)
    
    # Term 2: -(2*pi/3) * ux * uy * uz * (sum q_i * r_i)^2
    dipole_vec = -(2.0 * np.pi / 3.0) * ux * uy * uz * M_sq
    
    # Final Energy Correction [cite: 365, 366, 368, 369]
    e_total = e_3d + elc_corr + dipole_z + dipole_vec
    
    return e_total