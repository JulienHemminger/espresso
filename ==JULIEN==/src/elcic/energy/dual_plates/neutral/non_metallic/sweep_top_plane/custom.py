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
    lz = lz_full-gap
    
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

 
    lambda_threshold = lz/2

    # NOTE: theres a bug in the masks that causes an error-spike when part.z = lz/2 (exactly middle)
    # 1. Bottom interface: 0 <= z < lambda
    mask_bot = (ps_orig[:, 2] >= 0.0) & (ps_orig[:, 2] < lambda_threshold)
    # 2. Top interface: (lz - lambda) < z <= lz
    mask_top = (ps_orig[:, 2] > (lz - lambda_threshold)) & (ps_orig[:, 2] <= lz)    
    # 3. Middle region: lambda <= z <= (lz - lambda)
    mask_mid = False & (ps_orig[:, 2] >= lambda_threshold) & (ps_orig[:, 2] <= (lz - lambda_threshold))
 


    # 1. Near-Images for Top Interface
    ps_p1 = ps_orig[mask_top | mask_mid]
    ps_p1[:, 2] = 2 * (lz_full - gap) - ps_p1[:, 2]
    qs_p1 = qs_orig[mask_top | mask_mid]  * dt

    # 2. Near-Images for Bottom Interface
    ps_m1 = ps_orig[mask_bot | mask_mid]
    ps_m1[:, 2] = 2 * gap - ps_m1[:, 2]
    qs_m1 = qs_orig[mask_bot | mask_mid]  * db


    # 3. Base energy (original configuration)
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps, gap, lz_full)
    e_near_top = 0.0
    e_near_bot = 0.0


    # ---- TOP contribution ----
    if np.any(mask_top | mask_mid):
        ps_total_top = np.vstack([ps_orig, ps_p1])
        qs_total_top = np.concatenate([qs_orig, qs_p1])

        e_pm1_top = _get_config_energy(system, ps_p1, qs_p1, pref, eps, gap, lz_full)
        e_lt_top = _get_config_energy(system, ps_total_top, qs_total_top, pref, eps, gap, lz_full)

        e_near_top = 0.5 * (e_lt_top - e_pm1_top + e_l0)

    # ---- BOTTOM contribution ----
    if np.any(mask_bot | mask_mid):
        ps_total_bot = np.vstack([ps_orig, ps_m1])
        qs_total_bot = np.concatenate([qs_orig, qs_m1])

        e_pm1_bot = _get_config_energy(system, ps_m1, qs_m1, pref, eps, gap, lz_full)
        e_lt_bot = _get_config_energy(system, ps_total_bot, qs_total_bot, pref, eps, gap, lz_full)

        e_near_bot = 0.5 * (e_lt_bot - e_pm1_bot + e_l0)
    
    # Total near-field contribution
    e_near = e_near_top + e_near_bot

    # 3. Far-Field Energy (Top specific)
    e_far = 0#  pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    return e_near + e_far

