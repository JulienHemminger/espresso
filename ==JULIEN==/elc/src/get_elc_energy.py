import numpy as np
import espressomd
import espressomd.electrostatics

def get_elc_energy(p3m, gap_size, pw_error, system):
    """
    Computes the ELC-corrected electrostatic energy for a slab geometry,
    supporting both neutral and non-neutral systems.
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, pos = parts.q, parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Calculate global charge properties
    q_total = np.sum(qs)
    mz = np.sum(qs * zs)
    mz2 = np.sum(qs * zs**2)
    
    # 1. Standard 3D P3M energy calculation
    # Note: P3M must be configured to handle the neutralizing background.
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    prefactor = p3m.prefactor
    
    # 2. Corrected Dipole Term for Non-Neutral Systems (Eq. 3.8)
    # Replaces the standard dipole term to account for the background
    dipole_term = (2.0 * np.pi / (lx * ly * lz)) * (mz - (lz / 2.0) * q_total)**2
    e_corr_dipole = dipole_term * prefactor

    # 3. Dynamic Reciprocal Space Correction
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q, indexing='ij')
    P, Q = P.flatten(), Q.flatten()
    
    mask = (P != 0) | (Q != 0)
    P, Q = P[mask], Q[mask]
    
    fx, fy = P / lx, Q / ly
    f = np.sqrt(fx**2 + fy**2)
    
    valid = f <= f_max
    P, Q, f, fx, fy = P[valid], Q[valid], f[valid], fx[valid], fy[valid]

    # Compute Chi products
    arg_x = 2.0 * np.pi * fx[None, :] * xs[:, None]
    arg_y = 2.0 * np.pi * fy[None, :] * ys[:, None]
    
    cos_x, sin_x = np.cos(arg_x), np.sin(arg_x)
    cos_y, sin_y = np.cos(arg_y), np.sin(arg_y)
    
    arg_z = 2.0 * np.pi * f[None, :]
    exp_p = np.exp(zs[:, None] * arg_z)
    exp_m = np.exp(-zs[:, None] * arg_z)
    
    def get_sum(ez, cx, cy):
        return np.sum(qs[:, None] * ez * cx * cy, axis=0)

    s_cc_p, s_cc_m = get_sum(exp_p, cos_x, cos_y), get_sum(exp_m, cos_x, cos_y)
    s_sc_p, s_sc_m = get_sum(exp_p, sin_x, cos_y), get_sum(exp_m, sin_x, cos_y)
    s_cs_p, s_cs_m = get_sum(exp_p, cos_x, sin_y), get_sum(exp_m, cos_x, sin_y)
    s_ss_p, s_ss_m = get_sum(exp_p, sin_x, sin_y), get_sum(exp_m, sin_x, sin_y)

    chi_prod = (s_cc_p * s_cc_m + s_sc_p * s_sc_m + 
                s_cs_p * s_cs_m + s_ss_p * s_ss_m)
    
    replica_factor = np.exp(-2.0 * np.pi * f * lz) / (1.0 - np.exp(-2.0 * np.pi * f * lz))
    e_corr_recip = -np.sum(((1.0 / (lx * ly)) / f) * replica_factor * chi_prod)
    
    # 4. Additional Background Subtraction Term (Eq. 3.9)
    # Corrects for the interaction of the homogeneous background
    e_bg_sub = (2.0 * np.pi / (lx * ly)) * (
        - (1.0 / lz) * q_total * mz2 + 
        q_total * mz - 
        (1.0 / 6.0) * lz * (q_total**2)
    )
    
    return e_3d + e_corr_dipole + (prefactor * e_corr_recip) + (prefactor * e_bg_sub)