import numpy as np

def get_elc_energy(p3m, gap_size, pw_error, system):
    """
    Computes the total electrostatic energy of a 2D+h slab system using ELC.
    Handles both neutral and non-neutral systems by correcting the 3D P3M base.
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    
    # 1. Obtain the 3D Periodic Energy from P3M
    # Note: P3M usually includes a term -pi/(2*V*alpha^2) * Q_total^2 
    # and a dipole correction if metallic BCs are not used.
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    pre = p3m.prefactor
    
    # 2. Calculate Charge Moments
    xi0 = np.sum(qs)          # Total net charge
    xi1 = np.sum(qs * zs)     # Z-component of dipole moment
    xi2 = np.sum(qs * zs**2)  # Second moment of z-coordinates
    
    # 3. Handle the Non-Neutral/Dipole Corrections
    # We must "undo" the 3D dipole term and add the specific 2D slab correction.
    # The term below accounts for the 2D slab in a 3D periodic background.
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    
    # This term replaces the standard (mz^2) correction for non-neutral systems:
    # It represents the interaction of the slab with the neutralizing background
    # and the specific geometry of the slab dipole.
    e_non_neutral_corr = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)
    
    # 4. Reciprocal Space ELC Term (The Layer Correction)
    # Determine the number of modes based on the required pairwise error
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    
    # Exclude the k=0 mode (handled by the real space and dipole terms)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P/lx)**2 + (Q/ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    # Precompute trigonometric and exponential parts for the form factors
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    
    # Particle-wise components
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Compute form factors (Chi) linearly: O(N * N_modes)
    def s_term(ez, c1, c2): 
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Summing over the four combinations of sin/cos for the 2D Fourier transform
    chi = (s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy) + 
           s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy) +
           s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy) + 
           s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy))
    
    # Convergence factor for the infinite replication of layers in Z
    # This accounts for the gap between slabs
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    
    # The reciprocal energy correction
    e_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)
    
    # 5. Final Energy Assembly
    # Total = 3D Energy + Non-Neutral/Dipole Correction + Reciprocal ELC Correction
    return e_3d + (pre * e_non_neutral_corr) + (pre * e_recip)