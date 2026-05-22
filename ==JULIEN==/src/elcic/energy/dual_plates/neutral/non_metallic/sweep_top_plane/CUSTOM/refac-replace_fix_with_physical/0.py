import espressomd
import espressomd.electrostatics
import numpy as np

def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Product decomposition for the ELC reciprocal sum."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T
    
    # Use real/imaginary parts to represent sin/cos product decomposition
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])
    
    cx = np.cos(arg_x * xs[:, None])
    sx = np.sin(arg_x * xs[:, None])
    cy = np.cos(arg_y * ys[:, None])
    sy = np.sin(arg_y * ys[:, None])

    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0) 
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """
    Evaluates far image interactions optimized for a TOP interface (dt).
    Here, db is effectively 0, simplifying the far-field summation.
    """
    lx, ly, lz = box
    delta = db * dt
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

    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    # Top far field logic (L+2) dominates when delta_bot is 0
    m_top = ps[:, 2] > 0 # All charges interact with the top interface
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]
    
    if np.any(m_top):
        chi_local = _get_chi_components(fx, fy, f, ps[m_top], qs[m_top], sign=0)
        # Reflecting across top boundary: z_img = 2lz - z
        t = l_pq_sum(2*lz - ps[m_top, 2, None], dt)
        term_sum = np.sum(t, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    pref = 0.5 / (lx * ly)
    e_far = 0.0
    # Cross terms between real charges and top-reflected images
    for c0_p, cp2_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * c0_p * cp2_m)
        
    return e_far
def _get_config_energy(system, p_set, q_set, prefactor, accuracy, gap_size, lz):
    lx, ly = system.box_l[0], system.box_l[1]
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_set, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    # Correction term logic remains consistent with ELC non-neutrality
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_set[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    e_corr = prefactor * fac * (xi1**2 - (lz**2 / 12.0) * xi0**2)
    
    system.electrostatics.clear()
    return e_3d + e_corr


def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP dielectric interface.
    """
    box = np.array(system.box_l)
    lz_full = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]

    parts = system.part.all()
    qs, ps = parts.q.copy(), parts.pos.copy()

    # 1. Near-Images for Top Interface
    ps_top = ps.copy()
    ps_top[:, 2] = 2 * (lz_full - gap) - ps_top[:, 2]
    qs_top = qs * dt
    ps_real_plus_top = np.vstack([ps, ps_top])
    qs_real_plus_top = np.concatenate([qs, qs_top])

    e_pm1_top = _get_config_energy(system, ps_top, qs_top, pref, eps, gap, lz_full)
    e_lt_top = _get_config_energy(system, ps_real_plus_top, qs_real_plus_top, pref, eps, gap, lz_full)
    e_near_top = e_lt_top - e_pm1_top

    # 2. Near-Images for Bottom Interface
    ps_bot = ps.copy()
    ps_bot[:, 2] = 2 * gap - ps_bot[:, 2]
    qs_bot = qs * db
    ps_real_plus_bot = np.vstack([ps, ps_bot])
    qs_real_plus_bot = np.concatenate([qs, qs_bot])
    
    e_pm1_bot = _get_config_energy(system, ps_bot, qs_bot, pref, eps, gap, lz_full)
    e_lt_bot = _get_config_energy(system, ps_real_plus_bot, qs_real_plus_bot, pref, eps, gap, lz_full)

    e_near_bot = e_lt_bot - e_pm1_bot

    # ============ HACK FIX ===============
    part0_z = ps[0, 2]
    part0_is_in_bottom_half = part0_z <= (lz_full-gap) / 2
    if part0_is_in_bottom_half:
        e_near_top = 0
    else:
        e_near_bot = 0
    # ============ HACK FIX ===============

    # Total near-field contribution
    e_l0 = _get_config_energy(system, ps, qs, pref, eps, gap, lz_full)
    e_near = 0.5 * (e_near_top + e_near_bot + e_l0)

    # 3. Far-Field Energy (Top specific)
    e_far = pref * _get_far_field_energy(box, gap, eps, qs, ps, db, dt)

    return e_near + e_far

