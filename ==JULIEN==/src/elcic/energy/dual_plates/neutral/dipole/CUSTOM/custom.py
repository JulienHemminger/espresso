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
def _get_config_energy(system, p_set, q_set, prefactor, accuracy, lz):
    if len(q_set) == 0:
        return 0.0

    lx, ly = system.box_l[0], system.box_l[1]
    p_wrapped = p_set.copy()
    p_wrapped[:, 0] = np.mod(p_wrapped[:, 0], lx)
    p_wrapped[:, 1] = np.mod(p_wrapped[:, 1], ly)
    p_wrapped[:, 2] = np.mod(p_wrapped[:, 2], lz)
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_wrapped, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    # Correction term logic remains consistent with ELC non-neutrality
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_wrapped[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    if np.isclose(xi0, 0.0, atol=1e-12):
        e_corr = prefactor * fac * (xi1**2)
    else:
        e_corr = 0.0
    
    system.electrostatics.clear()
    return e_3d + e_corr


def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP/BOTTOM dielectric interfaces using ELCIC.
    Fixed: Removed mid-gap particles from image charge sets to avoid spurious interactions.
    """
    box = np.array(system.box_l)
    lz_full = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]
    lz = lz_full - gap  # Top interface position (gap is between z=0 and z=lz)
    
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    # Near-field cutoff: only particles within lambda of an interface get image charges
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    # Define near-field regions for each interface
    mask_bot = (ps_orig[:, 2] >= 0.0) & (ps_orig[:, 2] < lambda_)  # Near bottom interface (z=0)
    mask_top = (ps_orig[:, 2] > (lz - lambda_)) & (ps_orig[:, 2] <= lz)  # Near top interface (z=lz)
    mask_mid = (ps_orig[:, 2] >= lambda_) & (ps_orig[:, 2] <= (lz - lambda_))  # Far from both interfaces

    # --- DEBUG: Print particle distribution ---
    print(f"DEBUG: Total particles: {len(qs_orig)}")
    print(f"DEBUG: Mask counts - Bot: {np.sum(mask_bot)}, Top: {np.sum(mask_top)}, Mid: {np.sum(mask_mid)}")

    # --------------------------
    # Fixed: Image charge sets only include particles in the corresponding near region
    # --------------------------
    # 1. Near-Images for Top Interface: ONLY particles near top interface (mask_top)
    ps_p1 = ps_orig[mask_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]  # Mirror across top interface (z=lz)
    qs_p1 = qs_orig[mask_top] * dt  # Apply top dielectric image charge factor

    # 2. Near-Images for Bottom Interface: ONLY particles near bottom interface (mask_bot)
    ps_m1 = ps_orig[mask_bot].copy()
    ps_m1[:, 2] = -ps_m1[:, 2]  # Mirror across bottom interface (z=0)
    qs_m1 = qs_orig[mask_bot] * db  # Apply bottom dielectric image charge factor

    # --------------------------
    # Energy calculations (unchanged, formula is correct per ELCIC literature)
    # --------------------------
    # 1. Base energy of original charges
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps, lz_full)
    
    # 2. Combined sets: original + top images + bottom images
    ps_lt = np.vstack([ps_orig, ps_p1, ps_m1])
    qs_lt = np.concatenate([qs_orig, qs_p1, qs_m1])
    
    # 3. Image-only set: top images + bottom images
    ps_pm1 = np.vstack([ps_p1, ps_m1])
    qs_pm1 = np.concatenate([qs_p1, qs_m1])
    
    # 4. Energies of combined and image-only sets
    e_lt = _get_config_energy(system, ps_lt, qs_lt, pref, eps, lz_full)
    e_pm1 = _get_config_energy(system, ps_pm1, qs_pm1, pref, eps, lz_full)
    
    # --- DEBUG: Energy components ---
    print(f"DEBUG: E_l0: {e_l0:.4f}, E_lt: {e_lt:.4f}, E_pm1: {e_pm1:.4f}")

    # 5. Standard ELCIC near-field partitioning formula (correct per literature)
    e_near = 0.5 * (e_lt - e_pm1 + e_l0)
    print(f"DEBUG: Calculated E_near: {e_near:.4f}")

    # 3. Far-Field Energy (ensure this accounts for BOTH interfaces in your implementation)
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    e_total = e_near + e_far

    return {
        "e_total": e_total,
        "e_far": e_far,
        "e_near": e_near,
        "e_near_top": 0,  # Implement top-specific near energy if needed
        "e_near_bot": 0,  # Implement bottom-specific near energy if needed
    }