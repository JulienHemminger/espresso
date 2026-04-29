import espressomd
import espressomd.electrostatics
import numpy as np


def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    # --- 1. Setup P3M and Primary Layer (L0) ---
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T

    system.electrostatics.solver = p3m
    e_L0_3d = system.analysis.energy()["total"]

    # Charge Moments for L0
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    
    # ELC correction for L0 [cite: 91]
    e_elc_const_L0 = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    
    # --- 2. Reciprocal Space Setup ---
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))
    p, q = np.arange(-p_max, p_max + 1), np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx)**2 + (Q / ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f

    def get_chi_terms(q_vec, z_vec):
        cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
        cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
        ex_p, ex_m = np.exp(arg_z * z_vec[:, None]), np.exp(-arg_z * z_vec[:, None])
        
        def s(ez, c1, c2): return np.sum(q_vec[:, None] * ez * c1 * c2, axis=0)
        
        # Returns Chi components (plus and minus) 
        return {
            "p": [s(ex_p, cx, cy), s(ex_p, sx, cy), s(ex_p, cx, sy), s(ex_p, sx, sy)],
            "m": [s(ex_m, cx, cy), s(ex_m, sx, cy), s(ex_m, cx, sy), s(ex_m, sx, sy)]
        }

    chi_L0 = get_chi_terms(qs, zs)
    
    # Standard ELC factor for primary layer [cite: 76]
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    chi_val_L0 = sum(p_i * m_i for p_i, m_i in zip(chi_L0["p"], chi_L0["m"]))
    e_L0_elc_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi_val_L0)
    
    e_L0_elc = prefactor * (e_elc_const_L0 + e_L0_elc_recip)
    e_L0_total = e_L0_3d + e_L0_elc

    # --- 3. Subdivide L0 based on distance to surfaces (Eq 4.3) ---
    # L0,+1: charges near top interface (lz - gap_size < z <= lz)
    # L0,0: charges in middle (gap_size < z <= lz - gap_size)
    # L0,-1: charges near bottom interface (0 <= z <= gap_size)
    
    mask_top = zs > (lz - gap_size)
    mask_bot = zs < gap_size
    mask_mid = ~(mask_top | mask_bot)
    
    # --- 4. Compute L±1 (first generation images close to interfaces) ---
    # L+1: images from L0,+1 via first term of series (2.5)
    # L-1: images from L0,-1 via first term of series (2.3)
    
    qs_L_plus1 = delta_mid_top * qs[mask_top]
    zs_L_plus1 = 2 * lz - zs[mask_top]
    xs_L_plus1 = xs[mask_top]
    ys_L_plus1 = ys[mask_top]
    
    qs_L_minus1 = delta_mid_bot * qs[mask_bot]
    zs_L_minus1 = -zs[mask_bot]
    xs_L_minus1 = xs[mask_bot]
    ys_L_minus1 = ys[mask_bot]
    
    # Combine all charges for LT = L-1 ∪ L0 ∪ L+1
    qs_LT = np.concatenate([qs_L_minus1, qs, qs_L_plus1])
    xs_LT = np.concatenate([xs_L_minus1, xs, xs_L_plus1])
    ys_LT = np.concatenate([ys_L_minus1, ys, ys_L_plus1])
    zs_LT = np.concatenate([zs_L_minus1, zs, zs_L_plus1])
    
    # --- 5. Compute L±2 (higher generation images far from interfaces) ---
    # Using far formula for L2 interactions
    
    Delta = delta_mid_bot * delta_mid_top  # Δ = Δ_b * Δ_t
    
    def L_pq(z_vals, arg_z):
        """Geometric series factor (Eq 4.4)"""
        return Delta * np.exp(-arg_z * z_vals[:, None]) / (1.0 - Delta * np.exp(-4 * arg_z * lz))
    
    def I_z(z_vals):
        """Moment sum (Eq 4.5)"""
        return (1.0 / (1.0 - Delta)) * (z_vals + 2 * lz * Delta / (1.0 - Delta))
    
    # Compute chi factors for L-2 (Eq 4.6)
    def get_chi_L_minus2():
        chi_terms = {"cc": 0, "sc": 0, "cs": 0, "ss": 0}
        chi_0 = 0
        chi_1 = 0
        
        # L0,-1 contribution (series 2.3 and 2.4, excluding first element)
        if np.any(mask_bot):
            q_bot = qs[mask_bot]
            z_bot = zs[mask_bot]
            x_bot = xs[mask_bot]
            y_bot = ys[mask_bot]
            
            L1 = L_pq(2*lz + z_bot, arg_z)  # Series 2.3
            L2 = L_pq(2*lz - z_bot, arg_z)  # Series 2.4
            
            for i, (cx_name, cy_name) in enumerate([("c","c"), ("s","c"), ("c","s"), ("s","s")]):
                cx = np.cos(arg_x * x_bot[:, None]) if cx_name == "c" else np.sin(arg_x * x_bot[:, None])
                cy = np.cos(arg_y * y_bot[:, None]) if cy_name == "c" else np.sin(arg_y * y_bot[:, None])
                chi_terms[cx_name+cy_name] += np.sum(q_bot[:, None] * delta_mid_bot * (L1 + L2) * cx * cy, axis=0)
            
            chi_0 += np.sum(q_bot * delta_mid_bot / (1 - Delta) * (1 - Delta + Delta))
            chi_1 += np.sum(q_bot * delta_mid_bot / (1 - Delta) * (
                -I_z(2*lz + z_bot) - I_z(2*lz - z_bot)))
        
        # L0,0 and L0,+1 contribution (all terms of series 2.3 and 2.4)
        mask_mid_top = mask_mid | mask_top
        if np.any(mask_mid_top):
            q_mt = qs[mask_mid_top]
            z_mt = zs[mask_mid_top]
            x_mt = xs[mask_mid_top]
            y_mt = ys[mask_mid_top]
            
            L1 = L_pq(z_mt, arg_z)  # Series 2.3
            L2 = L_pq(2*lz - z_mt, arg_z)  # Series 2.4
            
            for i, (cx_name, cy_name) in enumerate([("c","c"), ("s","c"), ("c","s"), ("s","s")]):
                cx = np.cos(arg_x * x_mt[:, None]) if cx_name == "c" else np.sin(arg_x * x_mt[:, None])
                cy = np.cos(arg_y * y_mt[:, None]) if cy_name == "c" else np.sin(arg_y * y_mt[:, None])
                chi_terms[cx_name+cy_name] += np.sum(q_mt[:, None] * delta_mid_bot * (L1 + L2) * cx * cy, axis=0)
            
            chi_0 += np.sum(q_mt * delta_mid_bot / (1 - Delta))
            chi_1 += np.sum(q_mt * delta_mid_bot / (1 - Delta) * (-I_z(z_mt) - I_z(2*lz - z_mt)))
        
        return chi_terms, chi_0, chi_1
    
    # Compute chi factors for L+2 (Eq 4.10)
    def get_chi_L_plus2():
        chi_terms = {"cc": 0, "sc": 0, "cs": 0, "ss": 0}
        chi_0 = 0
        chi_1 = 0
        
        # L0,+1 contribution (series 2.5 and 2.6, excluding first element)
        if np.any(mask_top):
            q_top = qs[mask_top]
            z_top = zs[mask_top]
            x_top = xs[mask_top]
            y_top = ys[mask_top]
            
            L1 = L_pq(4*lz - z_top, arg_z)  # Series 2.5
            L2 = L_pq(2*lz + z_top, arg_z)  # Series 2.6
            
            for i, (cx_name, cy_name) in enumerate([("c","c"), ("s","c"), ("c","s"), ("s","s")]):
                cx = np.cos(arg_x * x_top[:, None]) if cx_name == "c" else np.sin(arg_x * x_top[:, None])
                cy = np.cos(arg_y * y_top[:, None]) if cy_name == "c" else np.sin(arg_y * y_top[:, None])
                chi_terms[cx_name+cy_name] += np.sum(q_top[:, None] * delta_mid_top * (L1 + L2) * cx * cy, axis=0)
            
            chi_0 += np.sum(q_top * delta_mid_top / (1 - Delta) * (1 - Delta + Delta))
            chi_1 += np.sum(q_top * delta_mid_top / (1 - Delta) * (
                I_z(4*lz - z_top) + I_z(2*lz + z_top)))
        
        # L0,0 and L0,-1 contribution (all terms of series 2.5 and 2.6)
        mask_mid_bot = mask_mid | mask_bot
        if np.any(mask_mid_bot):
            q_mb = qs[mask_mid_bot]
            z_mb = zs[mask_mid_bot]
            x_mb = xs[mask_mid_bot]
            y_mb = ys[mask_mid_bot]
            
            L1 = L_pq(2*lz - z_mb, arg_z)  # Series 2.5
            L2 = L_pq(2*lz + z_mb, arg_z)  # Series 2.6
            
            for i, (cx_name, cy_name) in enumerate([("c","c"), ("s","c"), ("c","s"), ("s","s")]):
                cx = np.cos(arg_x * x_mb[:, None]) if cx_name == "c" else np.sin(arg_x * x_mb[:, None])
                cy = np.cos(arg_y * y_mb[:, None]) if cy_name == "c" else np.sin(arg_y * y_mb[:, None])
                chi_terms[cx_name+cy_name] += np.sum(q_mb[:, None] * delta_mid_top * (L1 + L2) * cx * cy, axis=0)
            
            chi_0 += np.sum(q_mb * delta_mid_top / (1 - Delta))
            chi_1 += np.sum(q_mb * delta_mid_top / (1 - Delta) * (I_z(2*lz - z_mb) + I_z(2*lz + z_mb)))
        
        return chi_terms, chi_0, chi_1
    
    chi_L_minus2, chi0_L_minus2, chi1_L_minus2 = get_chi_L_minus2()
    chi_L_plus2, chi0_L_plus2, chi1_L_plus2 = get_chi_L_plus2()
    
    # Interaction L0 with L-2 (Eq 3.4)
    e_L0_L_minus2 = 0
    for key in ["cc", "sc", "cs", "ss"]:
        prod = chi_L0["m"][["cc", "sc", "cs", "ss"].index(key)] * chi_L_minus2[key]
        e_L0_L_minus2 += np.sum((1.0 / (lx * ly * f)) * prod)
    
    # Interaction L0 with L+2 (Eq 3.4)
    e_L0_L_plus2 = 0
    for key in ["cc", "sc", "cs", "ss"]:
        prod = chi_L0["p"][["cc", "sc", "cs", "ss"].index(key)] * chi_L_plus2[key]
        e_L0_L_plus2 += np.sum((1.0 / (lx * ly * f)) * prod)
    
    e_L2_total = -prefactor * (e_L0_L_minus2 + e_L0_L_plus2)
    
    # --- 6. Compute LT interaction using P3M+ELC (Eq 4.14) ---
    # Φ(L0,LT) = 1/2[Φ(LT,LT) - Φ(L±1,L±1) + Φ(L0,L0)]
    
    # Save original system state
    original_box_l = system.box_l.copy()
    
    # Create extended box for LT: Lz = lz + 3*gap_size (to leave gap on both sides)
    Lz = lz + 3 * gap_size
    
    # Clear particles and resize system
    system.part.clear()
    system.electrostatics.clear()
    system.box_l = [lx, ly, Lz]
    
    # Shift z-coordinates to account for gap below (gap_size offset from bottom)
    # This ensures gap_size space below L-1 and gap_size space above L+1
    z_offset = gap_size
    
    # Add LT particles (L-1 ∪ L0 ∪ L+1)
    N_LT = len(qs_LT)
    for i in range(N_LT):
        system.part.add(
            pos=[xs_LT[i], ys_LT[i], zs_LT[i] + z_offset],
            q=qs_LT[i]
        )
    
    # Compute Φ(LT,LT) using P3M + ELC
    p3m_LT = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m_LT
    e_LT_3d = system.analysis.energy()["total"]
    
    # ELC correction for LT (using non-neutral ELC formula from Eq 3.10)
    parts_LT = system.part.all()
    qs_LT_sys = parts_LT.q
    zs_LT_sys = parts_LT.pos[:, 2]
    
    # Charge moments for LT
    xi0_LT = np.sum(qs_LT_sys)
    xi1_LT = np.sum(qs_LT_sys * zs_LT_sys)
    xi2_LT = np.sum(qs_LT_sys * zs_LT_sys**2)
    
    volume_LT = lx * ly * Lz
    fac_LT = 2.0 * np.pi / volume_LT
    
    # Constant term (modified dipole term for non-neutral system, Eq 3.8)
    e_elc_const_LT = fac_LT * (xi1_LT**2 - xi0_LT * xi2_LT - (Lz**2 / 12.0) * xi0_LT**2)
    
    # Reciprocal space term (similar to L0 but for LT)
    # Recompute for new box size
    arg_z_LT = 2.0 * np.pi * f
    rep_LT = np.exp(-arg_z_LT * Lz) / (1.0 - np.exp(-arg_z_LT * Lz))
    
    # Get chi terms for LT
    def get_chi_terms_LT(q_vec, z_vec, x_vec, y_vec):
        cx = np.cos(arg_x * x_vec[:, None])
        sx = np.sin(arg_x * x_vec[:, None])
        cy = np.cos(arg_y * y_vec[:, None])
        sy = np.sin(arg_y * y_vec[:, None])
        ex_p = np.exp(arg_z_LT * z_vec[:, None])
        ex_m = np.exp(-arg_z_LT * z_vec[:, None])
        
        def s(ez, c1, c2): 
            return np.sum(q_vec[:, None] * ez * c1 * c2, axis=0)
        
        return {
            "p": [s(ex_p, cx, cy), s(ex_p, sx, cy), s(ex_p, cx, sy), s(ex_p, sx, sy)],
            "m": [s(ex_m, cx, cy), s(ex_m, sx, cy), s(ex_m, cx, sy), s(ex_m, sx, sy)]
        }
    
    chi_LT = get_chi_terms_LT(qs_LT_sys, zs_LT_sys, parts_LT.pos[:, 0], parts_LT.pos[:, 1])
    chi_val_LT = sum(p_i * m_i for p_i, m_i in zip(chi_LT["p"], chi_LT["m"]))
    e_LT_elc_recip = -np.sum((1.0 / (lx * ly * f)) * rep_LT * chi_val_LT)
    
    e_LT_elc = prefactor * (e_elc_const_LT + e_LT_elc_recip)
    e_LT_LT_total = e_LT_3d + e_LT_elc
    
    # Compute Φ(L±1,L±1)
    system.part.clear()
    system.electrostatics.clear()
    
    # Add only L±1 particles
    qs_L1 = np.concatenate([qs_L_minus1, qs_L_plus1])
    xs_L1 = np.concatenate([xs_L_minus1, xs_L_plus1])
    ys_L1 = np.concatenate([ys_L_minus1, ys_L_plus1])
    zs_L1 = np.concatenate([zs_L_minus1, zs_L_plus1])
    
    N_L1 = len(qs_L1)
    for i in range(N_L1):
        system.part.add(
            pos=[xs_L1[i], ys_L1[i], zs_L1[i] + z_offset],
            q=qs_L1[i]
        )
    
    if N_L1 > 0:
        p3m_L1 = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
        )
        system.electrostatics.solver = p3m_L1
        e_L1_3d = system.analysis.energy()["total"]
        
        # ELC correction for L1
        parts_L1 = system.part.all()
        qs_L1_sys = parts_L1.q
        zs_L1_sys = parts_L1.pos[:, 2]
        
        xi0_L1 = np.sum(qs_L1_sys)
        xi1_L1 = np.sum(qs_L1_sys * zs_L1_sys)
        xi2_L1 = np.sum(qs_L1_sys * zs_L1_sys**2)
        
        e_elc_const_L1 = fac_LT * (xi1_L1**2 - xi0_L1 * xi2_L1 - (Lz**2 / 12.0) * xi0_L1**2)
        
        chi_L1 = get_chi_terms_LT(qs_L1_sys, zs_L1_sys, parts_L1.pos[:, 0], parts_L1.pos[:, 1])
        chi_val_L1 = sum(p_i * m_i for p_i, m_i in zip(chi_L1["p"], chi_L1["m"]))
        e_L1_elc_recip = -np.sum((1.0 / (lx * ly * f)) * rep_LT * chi_val_L1)
        
        e_L1_elc = prefactor * (e_elc_const_L1 + e_L1_elc_recip)
        e_L1_L1_total = e_L1_3d + e_L1_elc
    else:
        e_L1_L1_total = 0.0
        e_L1_3d = 0.0
        e_L1_elc = 0.0
    
    # Φ(L0,L0) was already computed as e_L0_total
    
    # Final LT contribution (Eq 4.14)
    e_LT_contribution = 0.5 * (e_LT_LT_total - e_L1_L1_total + e_L0_total)
    
    # Restore original system state
    system.part.clear()
    system.electrostatics.clear()
    system.box_l = original_box_l
    
    # Restore original particles
    for i in range(len(qs)):
        system.part.add(pos=[xs[i], ys[i], zs[i]], q=qs[i])
    
    # Restore original P3M solver
    system.electrostatics.solver = p3m
    
    return {
        "e_near": 0.0,
        "e_far": float(e_L0_total + e_L2_total + e_LT_contribution),
        "l0": {"e_3d": float(e_L0_3d), "e_elc": float(e_L0_elc), "total": float(e_L0_total)},
        "pm1": {"e_3d": 0.0, "e_elc": 0.0, "total": 0.0},
        "lt": {
            "e_3d": float(e_LT_3d), 
            "e_elc": float(e_LT_elc), 
            "total": float(e_LT_contribution),
            "e_LT_LT": float(e_LT_LT_total),
            "e_L1_L1": float(e_L1_L1_total)
        },
        "l1": {"e_3d": float(e_L1_3d), "e_elc": float(e_L1_elc), "total": float(e_L1_L1_total)},
        "l2": {"total": float(e_L2_total)},
        "e_far_detail": {}
    }

def get_elcic_energy(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top):
    contribs = get_elcic_energy_contribs(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top)
    return contribs["e_near"] + contribs["e_far"]