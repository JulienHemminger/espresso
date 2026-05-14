import espressomd
import espressomd.electrostatics
import numpy as np

def _get_non_neutral_correction(box, zs, qs):
    """Equation 3.10 & 3.11: Dipole and non-neutrality energy correction."""
    lx, ly, lz = box
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    fac = 2.0 * np.pi / (lx * ly * lz)
    return fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Equation 3.3: Product decomposition for the ELC reciprocal sum."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T
    
    if sign != 0:
        ez = np.exp(sign * 1j * arg_z * zs[:, None])
    else:
        ez = 1.0
    
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    # Returns [CC, SC, CS, SS]
    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0) 
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """Evaluates Equations 4.6 through 4.13 for far image interactions."""
    lx, ly, lz = box
    delta = db * dt
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    # Generate reciprocal space grid
    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    # Pre-calculate base Chi factors with imaginary exponentials
    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        """Proper geometric series sum for L_pq."""
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    def compute_field_contribution(mask, d_coeff, z_transform):
        if not np.any(mask):
            return [np.zeros_like(f) for _ in range(4)]
        
        chi_local = _get_chi_components(fx, fy, f, ps[mask], qs[mask], sign=0)
        z_transformed = z_transform(ps[mask, 2, None])
        
        # For particles in the gap region
        term1 = l_pq_sum(2 * lz + z_transformed, d_coeff)
        term2 = l_pq_sum(2 * lz - z_transformed, delta)
        
        combined_term = np.sum(term1 + term2, axis=0)
        return [c * combined_term for c in chi_local]

    # Bottom (L-2) and Top (L+2) far field logic
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)
    
    # Chi_-2^+ components (Eq 4.8 and 4.9)
    chi_m2_p_bot = compute_field_contribution(m_bot, db, lambda z: z)
    chi_m2_p_top = compute_field_contribution(~m_bot, 1.0, lambda z: -2*lz + z)
    chi_m2_p = [a + b for a, b in zip(chi_m2_p_bot, chi_m2_p_top)]
    
    # Chi_+2^- components (Eq 4.10 and 4.11)
    chi_p2_m_top = compute_field_contribution(m_top, dt, lambda z: lz - z)
    chi_p2_m_bot = compute_field_contribution(~m_top, 1.0, lambda z: -lz - z)
    chi_p2_m = [a + b for a, b in zip(chi_p2_m_top, chi_p2_m_bot)]

    # Final Energy assembly (Eq 4.6 and 4.7)
    pref = 1.0 / (lx * ly)
    
    # Real part extraction for energy
    e_far = 0.0
    for chi_m, chi_p in zip(chi0_m, chi_m2_p):
        e_far += pref * np.sum((1.0 / f) * np.real(np.conj(chi_m) * chi_p))
    
    for chi_p, chi_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * np.real(np.conj(chi_p) * chi_m))
    
    return e_far

def _get_config_energy(system, p_set, q_set, prefactor, accuracy):
    """Helper to swap system particles and compute P3M energy without correction."""
    system.part.clear()
    system.part.add(pos=p_set, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    e_3d = system.analysis.energy()["total"]
    system.electrostatics.clear()
    
    return e_3d

def get_elcic_energy(system, params: dict):
    """
    Main ELCIC energy calculation.
    
    BUG FIXES:
    1. Corrected near-field linear combination to match C++ implementation
    2. Fixed image charge treatment for dielectric contrast
    3. Added proper self-energy correction for dielectric layers
    4. Fixed non-neutrality correction application
    5. Corrected image position calculations
    """
    box = system.box_l
    lz = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]

    # Cache original state
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    # Calculate box_h (height of simulation region)
    box_h = lz - gap
    
    # Space layer calculation (1/3 of gap, but check constraints)
    space_layer = gap / 3.0
    
    # Check if dielectric contrast is on
    dielectric_contrast_on = (db != 0.0 or dt != 0.0)
    
    # 1. Generate Near-Image Coordinates
    m_bot = ps_orig[:, 2] < space_layer
    m_top = ps_orig[:, 2] > (box_h - space_layer)
    
    ps_m1 = ps_orig[m_bot].copy()
    ps_m1[:, 2] *= -1  # Mirror at z=0
    
    ps_p1 = ps_orig[m_top].copy()
    ps_p1[:, 2] = 2 * box_h - ps_p1[:, 2]  # Mirror at z=2*box_h

    qs_m1 = qs_orig[m_bot] * db
    qs_p1 = qs_orig[m_top] * dt

    # 2. Compute energy components following C++ logic
    # E_original: energy with original charges only
    e_original = _get_config_energy(system, ps_orig, qs_orig, pref, eps)
    
    # Apply non-neutrality correction to original configuration
    e_corr_original = pref * _get_non_neutral_correction(box, ps_orig[:, 2], qs_orig)
    e_original += e_corr_original
    
    if not dielectric_contrast_on:
        # No dielectric contrast: just use original energy
        e_near = e_original
    else:
        # E_both: energy with original + image charges
        if len(qs_m1) > 0 and len(qs_p1) > 0:
            ps_both = np.vstack([ps_orig, ps_m1, ps_p1])
            qs_both = np.concatenate([qs_orig, qs_m1, qs_p1])
        elif len(qs_m1) > 0:
            ps_both = np.vstack([ps_orig, ps_m1])
            qs_both = np.concatenate([qs_orig, qs_m1])
        elif len(qs_p1) > 0:
            ps_both = np.vstack([ps_orig, ps_p1])
            qs_both = np.concatenate([qs_orig, qs_p1])
        else:
            ps_both = ps_orig
            qs_both = qs_orig
        
        e_both = _get_config_energy(system, ps_both, qs_both, pref, eps)
        e_corr_both = pref * _get_non_neutral_correction(box, ps_both[:, 2], qs_both)
        e_both += e_corr_both
        
        # E_image: energy with image charges only
        if len(qs_m1) > 0 and len(qs_p1) > 0:
            ps_image = np.vstack([ps_m1, ps_p1])
            qs_image = np.concatenate([qs_m1, qs_p1])
        elif len(qs_m1) > 0:
            ps_image = ps_m1
            qs_image = qs_m1
        elif len(qs_p1) > 0:
            ps_image = ps_p1
            qs_image = qs_p1
        else:
            ps_image = np.zeros((0, 3))
            qs_image = np.zeros(0)
        
        if len(qs_image) > 0:
            e_image = _get_config_energy(system, ps_image, qs_image, pref, eps)
            e_corr_image = pref * _get_non_neutral_correction(box, ps_image[:, 2], qs_image)
            e_image += e_corr_image
        else:
            e_image = 0.0
        
        # Dielectric self-energy correction
        # This accounts for the interaction of images with the dielectric boundaries
        xy_area_inv = 1.0 / (box[0] * box[1])
        pref_di = pref * 2.0 * np.pi * xy_area_inv
        
        delta = db * dt
        shift = lz / 2.0
        
        # Collect moments for self-energy
        sum_q = np.sum(qs_orig)
        sum_qz = np.sum(qs_orig * (ps_orig[:, 2] - shift))
        sum_q_image = 0.0
        sum_qz_image = 0.0
        
        if delta != 0.0:
            fac_delta_mid_bot = db / (1.0 - delta)
            fac_delta_mid_top = dt / (1.0 - delta)
            fac_delta = delta / (1.0 - delta)
            
            for i, (q, z) in enumerate(zip(qs_orig, ps_orig[:, 2])):
                if z < space_layer:
                    sum_q_image += fac_delta * (db + 1.0) * q
                    # Image sum contribution for bottom
                    sum_qz_image += q * (fac_delta_mid_bot * db * delta * (-2.0 * box_h - z - shift) / (1.0 - delta) +
                                        fac_delta_mid_bot * delta * (-2.0 * box_h + z - shift) / (1.0 - delta))
                else:
                    sum_q_image += fac_delta_mid_bot * (1.0 + dt) * q
                    sum_qz_image += q * (fac_delta_mid_bot * (-z - shift) / (1.0 - delta) +
                                        fac_delta_mid_bot * delta * (-2.0 * box_h + z - shift) / (1.0 - delta))
                
                if z > (box_h - space_layer):
                    sum_q_image -= fac_delta * (dt + 1.0) * q
                    sum_qz_image -= q * (fac_delta_mid_top * dt * delta * (4.0 * box_h - z - shift) / (1.0 - delta) +
                                        fac_delta_mid_top * delta * (2.0 * box_h + z - shift) / (1.0 - delta))
                else:
                    sum_q_image -= fac_delta_mid_top * (1.0 + db) * q
                    sum_qz_image -= q * (fac_delta_mid_top * (2.0 * box_h - z - shift) / (1.0 - delta) +
                                        fac_delta_mid_top * delta * (2.0 * box_h + z - shift) / (1.0 - delta))
        
        e_self = -pref_di * (sum_qz * sum_q_image - sum_q * sum_qz_image)
        
        # Following C++ formula: 0.5 * (E_original + E_self + E_both - E_image)
        e_near = 0.5 * (e_original + e_self + e_both - e_image)

    # 3. Far-Field Correction
    if dielectric_contrast_on:
        e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)
    else:
        e_far = 0.0

    # Restore original state
    system.part.clear()
    system.part.add(pos=ps_orig, q=qs_orig)

    return e_near + e_far