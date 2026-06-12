import espressomd
import espressomd.electrostatics
import numpy as np

def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Decompose reciprocal sum into sin/cos product components for ELC."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T
    
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0) 
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """Computes far-field image interactions for dielectric interfaces."""
    lx, ly, lz = box
    delta = db * dt
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    # Generate reciprocal lattice vectors up to cutoff
    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    P, Q = np.meshgrid(p_range, q_range)
    mask = (P.flatten() != 0) | (Q.flatten() != 0)
    
    fx, fy = P.flatten()[mask] / lx, Q.flatten()[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    # Top interface interaction logic
    m_top = ps[:, 2] > 0
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]
    
    if np.any(m_top):
        chi_local = _get_chi_components(fx, fy, f, ps[m_top], qs[m_top], sign=0)
        t = l_pq_sum(2*lz - ps[m_top, 2, None], dt)
        term_sum = np.sum(t, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    pref = 0.5 / (lx * ly)
    return pref * sum(np.sum((1.0 / f) * c0_p * cp2_m) for c0_p, cp2_m in zip(chi0_p, chi_p2_m))

def _get_config_energy(system, p_set, q_set, prefactor, accuracy, lz):
    """Calculates electrostatic energy for a specific charge configuration."""
    if len(q_set) == 0: return 0.0

    # Reset system for P3M calculation
    lx, ly = system.box_l[0], system.box_l[1]
    p_wrapped = np.mod(p_set, [lx, ly, lz])
    system.part.clear()
    system.part.add(pos=p_wrapped, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    # Correction for potential non-neutrality in ELC
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_wrapped[:, 2])
    e_corr = prefactor * (2.0 * np.pi / (lx * ly * lz)) * (xi1**2) if np.isclose(xi0, 0.0, atol=1e-12) else 0.0
    
    system.electrostatics.clear()
    return e_3d + e_corr

def get_elcic_energy(system, params: dict):
    """Computes total electrostatic energy for 2D+h systems with dielectric interfaces."""
    box = np.array(system.box_l)
    lz_full, gap = box[2], params["gap_size"]
    pref, eps = params.get("prefactor", 1.0), params["pw_error"]
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]
    lz = lz_full - gap 
    
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    # Partition particles by distance to interfaces
    mask_bot = (ps_orig[:, 2] >= 0.0) & (ps_orig[:, 2] < lambda_)
    mask_top = (ps_orig[:, 2] > (lz - lambda_)) & (ps_orig[:, 2] <= lz)

    # Generate image charges near boundaries
    ps_p1, qs_p1 = ps_orig[mask_top].copy(), qs_orig[mask_top] * dt
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]
    ps_m1, qs_m1 = ps_orig[mask_bot].copy(), qs_orig[mask_bot] * db
    ps_m1[:, 2] = -ps_m1[:, 2]

    # Calculate near-field energy via ELCIC partitioning
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps, lz_full)
    e_lt = _get_config_energy(system, np.vstack([ps_orig, ps_p1, ps_m1]), np.concatenate([qs_orig, qs_p1, qs_m1]), pref, eps, lz_full)
    e_pm1 = _get_config_energy(system, np.vstack([ps_p1, ps_m1]), np.concatenate([qs_p1, qs_m1]), pref, eps, lz_full)
    e_near = 0.5 * (e_lt - e_pm1 + e_l0)

    # Add far-field contributions
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    return {
        "e_total": e_near + e_far,
        "e_far": e_far,
        "e_near": e_near,
        "e_near_top": 0, 
        "e_near_bot": 0 
    }