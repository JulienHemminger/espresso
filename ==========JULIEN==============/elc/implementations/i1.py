import numpy as np

def get_elc_energy(p3m, gap_size, pw_error, system):
    lx, ly, lz = system.box_l
    ux, uy, uz = 1.0/lx, 1.0/ly, 1.0/lz
    
    parts = system.part.all()
    qs = parts.q
    pos = parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # 1. Get the baseline 3D periodic energy
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    
    # 2. Calculate Dipole Correction [cite: 526, 528]
    # We subtract the 3D dipole term and add the 2D slab dipole term.
    # For a neutral system, this simplifies to the following z-component term:
    mu_z = np.sum(qs * zs)
    e_dipole = 2 * np.pi * ux * uy * uz * (mu_z**2)
    
    # 3. Calculate Fourier Correction (Far Formula) [cite: 512, 526]
    e_fourier = 0.0
    # The number of modes p,q depends on the required accuracy (pw_error)
    # and the gap_size (h). In practice, this is a tuning parameter[cite: 514, 516].
    # For this implementation, we sum over a standard range.
    max_pq = 10 
    
    for p in range(-max_pq, max_pq + 1):
        for q in range(-max_pq, max_pq + 1):
            if p == 0 and q == 0:
                continue
            
            # Frequency components [cite: 513]
            fpq = np.sqrt((ux * p)**2 + (uy * q)**2)
            omega_p = 2 * np.pi * ux * p
            omega_q = 2 * np.pi * uy * q
            
            # Product decomposition factors chi (eq 3.3) [cite: 518]
            # chi_plus uses exp(+2*pi*fpq*z), chi_minus uses exp(-2*pi*fpq*z)
            arg_x = omega_p * xs
            arg_y = omega_q * ys
            exp_pos = np.exp(2 * np.pi * fpq * zs)
            exp_neg = np.exp(-2 * np.pi * fpq * zs)
            
            cos_x, sin_x = np.cos(arg_x), np.sin(arg_x)
            cos_y, sin_y = np.cos(arg_y), np.sin(arg_y)
            
            # chi factors for the primary layer
            chi_p_cc = np.sum(qs * exp_pos * cos_x * cos_y)
            chi_p_sc = np.sum(qs * exp_pos * sin_x * cos_y)
            chi_p_cs = np.sum(qs * exp_pos * cos_x * sin_y)
            chi_p_ss = np.sum(qs * exp_pos * sin_x * sin_y)
            
            chi_m_cc = np.sum(qs * exp_neg * cos_x * cos_y)
            chi_m_sc = np.sum(qs * exp_neg * sin_x * cos_y)
            chi_m_cs = np.sum(qs * exp_neg * cos_x * sin_y)
            chi_m_ss = np.sum(qs * exp_neg * sin_x * sin_y)
            
            # Replicated layer factors X (eq 3.6 & 3.7) [cite: 527]
            # L_prime accounts for the infinite summation over replicas
            l_prime = 1.0 / (1.0 - np.exp(-4 * np.pi * lz * fpq))
            
            # Combine to get the correction for this mode [cite: 526]
            term = (chi_m_cc * chi_p_cc + chi_m_sc * chi_p_sc + 
                    chi_m_cs * chi_p_cs + chi_m_ss * chi_p_ss)
            
            e_fourier -= (ux * uy / (2 * fpq)) * term * l_prime * np.exp(-2 * np.pi * fpq * lz)

    # Total energy = E_3D + E_dipole + E_fourier [cite: 526]
    return e_3d + e_dipole + e_fourier