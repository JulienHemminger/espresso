import espressomd
import espressomd.electrostatics
import numpy as np

def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Equation 3.3: Product decomposition for the ELC reciprocal sum[cite: 118]."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T
    
    # Use real and imaginary parts to represent the sin/cos combinations [cite: 118]
    # chi = sum(q * exp(sign * 2pi * f * z) * exp(i * 2pi * (px/lx + qy/ly)))
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])
    exy = np.exp(1j * (arg_x * xs[:, None] + arg_y * ys[:, None]))
    
    res = np.sum(qs[:, None] * ez * exy, axis=0)
    # Decompose into CC, SC, CS, SS components if needed, 
    # but complex multiplication handles Eq. 3.4 more efficiently.
    return res

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """Evaluates Equation 3.4 for far image interactions[cite: 124]."""
    lx, ly, lz = box
    delta = db * dt
    # The cutoff frequency depends on gap_size (lambda) [cite: 113]
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    # Eq 4.4: L_pq geometric series sum [cite: 244]
    def l_pq_sum(z_dist, delta_coeff):
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * np.exp(-2.0 * np.pi * f * z_dist) / denom

    # Calculating Chi factors for far groups L-2 and L+2 using Eq 4.6/4.10 [cite: 259, 280]
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)
    m_mid = ~(m_bot | m_top)

    # chi_local is essentially the sum of charges with the xy-Fourier factor
    def get_chi_xy(mask):
        if not np.any(mask): return 0.0
        arg_x, arg_y = 2.0 * np.pi * fx, 2.0 * np.pi * fy
        return np.sum(qs[mask, None] * np.exp(1j * (arg_x * ps[mask, 0, None] + arg_y * ps[mask, 1, None])), axis=0)

    # L-2 contributions (Eq 4.6)
    chi_m2 = (get_chi_xy(m_bot) * (db * delta * l_pq_sum(2*lz + ps[m_bot, 2, None], 1.0).sum(axis=0) + 
                                  delta * l_pq_sum(2*lz - ps[m_bot, 2, None], 1.0).sum(axis=0)) +
              get_chi_xy(~m_bot) * (db * l_pq_sum(ps[~m_bot, 2, None], 1.0).sum(axis=0) + 
                                   delta * l_pq_sum(2*lz - ps[~m_bot, 2, None], 1.0).sum(axis=0)))

    # L+2 contributions (Eq 4.10)
    chi_p2 = (get_chi_xy(m_top) * (dt * delta * l_pq_sum(3*lz - ps[m_top, 2, None], 1.0).sum(axis=0) + 
                                  delta * l_pq_sum(2*lz + ps[m_top, 2, None], 1.0).sum(axis=0)) +
              get_chi_xy(~m_top) * (dt * l_pq_sum(2*lz - ps[~m_top, 2, None], 1.0).sum(axis=0) + 
                                   delta * l_pq_sum(2*lz + ps[~m_top, 2, None], 1.0).sum(axis=0)))

    # chi0 for the real particles
    chi0 = _get_chi_components(fx, fy, f, ps, qs, sign=0)
    
    # Eq 3.4 Prefactor is 1/(2 * lx * ly) [cite: 124]
    pref = 0.5 / (lx * ly)
    return pref * np.sum((1.0 / f) * np.real(chi0 * np.conj(chi_m2 + chi_p2)))

def _get_non_neutral_correction(box, zs, qs):
    """Equation 3.10: Corrected non-neutrality energy correction[cite: 185]."""
    lx, ly, Lz = box 
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    
    # Dipole term for ELC [cite: 185]
    fac = 2.0 * np.pi / (lx * ly * Lz)
    return fac * (xi1**2 - xi0 * xi2 - (Lz**2 / 12.0) * xi0**2)

def _get_config_energy(system, p_set, q_set, prefactor, accuracy, gap_size):
    """Applies Section IV.B artificial simulation box logic."""
    lx, ly, lz_phys = system.box_l[0], system.box_l[1], system.box_l[2]
    # Correct extended height per Section IV.B 
    extended_lz = lz_phys + 3.0 * gap_size
    
    system.part.clear()
    system.box_l = [lx, ly, extended_lz]
    system.part.add(pos=p_set, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(prefactor=prefactor, accuracy=accuracy, 
                                        check_neutrality=False, verbose=False)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    # Subtract 3D dipole and add ELC dipole term [cite: 154, 185]
    e_corr = prefactor * _get_non_neutral_correction(system.box_l, p_set[:, 2], q_set)
    
    system.electrostatics.clear()
    system.part.clear()
    system.box_l = [lx, ly, lz_phys]
    
    return e_3d + e_corr

def get_elcic_energy(system, params: dict):
    box = np.array(system.box_l)
    lz = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]

    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    # 1. Generate Near-Images (L-1 and L+1) [cite: 228, 235]
    m_bot = ps_orig[:, 2] <= gap
    m_top = ps_orig[:, 2] > (lz - gap)
    
    ps_m1 = ps_orig[m_bot].copy(); ps_m1[:, 2] *= -1
    ps_p1 = ps_orig[m_top].copy(); ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]

    qs_m1 = qs_orig[m_bot] * db
    qs_p1 = qs_orig[m_top] * dt

    # 2. Equation 4.14: Linear combination for Near-Field [cite: 299]
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps, gap)
    
    if len(qs_m1) > 0 or len(qs_p1) > 0:
        ps_images = np.vstack([p for p in [ps_m1, ps_p1] if len(p) > 0])
        qs_images = np.concatenate([q for q in [qs_m1, qs_p1] if len(q) > 0])
        
        e_pm1 = _get_config_energy(system, ps_images, qs_images, pref, eps, gap)
        e_lt = _get_config_energy(system, np.vstack([ps_orig, ps_images]), 
                                  np.concatenate([qs_orig, qs_images]), pref, eps, gap)
        e_near = 0.5 * (e_lt - e_pm1 + e_l0)
    else:
        e_near = e_l0

    # 3. Far-Field Correction (Eq. 4.6 - 4.13) [cite: 213, 239]
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    system.part.add(pos=ps_orig, q=qs_orig)
    return e_near + e_far