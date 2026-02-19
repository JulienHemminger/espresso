import numpy as np
import matplotlib.pyplot as plt
import espressomd
import espressomd.electrostatics

def get_elc_energy(p3m, gap_size, pw_error, system):
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, pos = parts.q, parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    prefactor = p3m.prefactor
    vol = lx * ly * lz

    # 1. Dipole Term (Constant time)
    mx, my, mz = np.sum(qs * xs), np.sum(qs * ys), np.sum(qs * zs)
    e_corr_dipole = (2.0 * np.pi / vol) * (mz**2 - (mx**2 + my**2 + mz**2) / 3.0) * prefactor

    # 2. Vectorized Reciprocal Correction
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_range = np.arange(-int(np.ceil(f_max * lx)), int(np.ceil(f_max * lx)) + 1)
    q_range = np.arange(-int(np.ceil(f_max * ly)), int(np.ceil(f_max * ly)) + 1)
    
    # Create a grid of (p, q) frequencies
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()
    
    # Remove (0,0) and filter by f_max
    mask = (P != 0) | (Q != 0)
    P, Q = P[mask], Q[mask]
    fx, fy = P / lx, Q / ly
    f = np.sqrt(fx**2 + fy**2)
    valid = f <= f_max
    P, Q, f, fx, fy = P[valid], Q[valid], f[valid], fx[valid], fy[valid]

    # Pre-calculate particle factors to avoid redundant loops
    # Use broadcasting: (N_particles, N_frequencies)
    omega_p = 2.0 * np.pi * fx
    omega_q = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f

    # Vectorized Chi products
    # Shape: (N_particles, N_freq)
    expp = np.exp(zs[:, None] * arg_z[None, :])
    expm = np.exp(-zs[:, None] * arg_z[None, :])
    cos_x = np.cos(xs[:, None] * omega_p[None, :])
    sin_x = np.sin(xs[:, None] * omega_p[None, :])
    cos_y = np.cos(ys[:, None] * omega_q[None, :])
    sin_y = np.sin(ys[:, None] * omega_q[None, :])

    # This part replaces the get_chi_product function with matrix multiplications
    def calc_s(ez, cx, cy):
        return np.sum(qs[:, None] * ez * cx * cy, axis=0)

    s_cc_p, s_cc_m = calc_s(expp, cos_x, cos_y), calc_s(expm, cos_x, cos_y)
    s_sc_p, s_sc_m = calc_s(expp, sin_x, cos_y), calc_s(expm, sin_x, cos_y)
    s_cs_p, s_cs_m = calc_s(expp, cos_x, sin_y), calc_s(expm, cos_x, sin_y)
    s_ss_p, s_ss_m = calc_s(expp, sin_x, sin_y), calc_s(expm, sin_x, sin_y)

    chi_prod = s_cc_p*s_cc_m + s_sc_p*s_sc_m + s_cs_p*s_cs_m + s_ss_p*s_ss_m
    replica_factor = np.exp(-2.0 * np.pi * f * lz) / (1.0 - np.exp(-2.0 * np.pi * f * lz))
    
    e_corr_recip = -np.sum(((1.0/lx) * (1.0/ly) / f) * replica_factor * chi_prod)
    
    return e_3d + e_corr_dipole + prefactor * e_corr_recip