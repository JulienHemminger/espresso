import espressomd
import espressomd.electrostatics
import numpy as np

import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy_contribs(
    system, total_gap, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Parameters:
    -----------
    total_gap : float
        The total empty space added to the box (h in some papers).
        We define lambda = total_gap / 3 as per Eq 4.14 logic.
    """
    # Define lambda as 1/3 of the total gap provided
    lam = total_gap / 3.0
    
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    N = len(qs)

    # --- 1. Subdivide L0 based on lambda ---
    # L0,-1: near bottom [0, lam]
    # L0,0:  middle [lam, lz - lam]
    # L0,+1: near top [lz - lam, lz]
    mask_L0_minus1 = (zs <= lam)
    mask_L0_plus1  = (zs >= lz - lam)
    mask_L0_0      = (zs > lam) & (zs < lz - lam)

    # --- 2. Generate Image Charges for L-1 and L+1 ---
    # L-1 (images of L0,-1 across bottom interface at z=0)
    q_L_minus1 = delta_mid_bot * qs[mask_L0_minus1]
    z_L_minus1 = -zs[mask_L0_minus1]
    
    # L+1 (images of L0,+1 across top interface at z=lz)
    q_L_plus1 = delta_mid_top * qs[mask_L0_plus1]
    z_L_plus1 = 2 * lz - zs[mask_L0_plus1]

    # --- 3. Construct the Expanded Box ---
    # New box height is lz + 3*lambda (which is lz + total_gap)
    Lz_expanded = lz + total_gap
    # We shift all charges by lam to center L0 and leave space for L-1 and L+1
    shift = lam

    # LT = L-1 U L0 U L+1
    q_LT = np.concatenate([q_L_minus1, qs, q_L_plus1])
    z_LT = np.concatenate([z_L_minus1 + shift, zs + shift, z_L_plus1 + shift])
    x_LT = np.concatenate([xs[mask_L0_minus1], xs, xs[mask_L0_plus1]])
    y_LT = np.concatenate([ys[mask_L0_minus1], ys, ys[mask_L0_plus1]])

    # --- 4. Energy Calculations using P3M + ELC ---
    # Save state
    original_box = system.box_l.copy()
    p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False
        )
    
    def get_layer_energy(q_arr, x_arr, y_arr, z_arr):
        if len(q_arr) == 0: return 0.0
        system.part.clear()
        system.box_l = [lx, ly, Lz_expanded]
        for i in range(len(q_arr)):
            system.part.add(pos=[x_arr[i], y_arr[i], z_arr[i]], q=q_arr[i])
        
        # P3M Setup
        system.electrostatics.solver = p3m
        
        e_3d = system.analysis.energy()["total"]
        # ELC correction (manually calculated or using system.electrostatics.ELC if preferred)
        e_elc = _calculate_elc_correction(
            system, q_arr, x_arr, y_arr, z_arr, lx, ly, Lz_expanded, lam, prefactor, p3m
        )
        return e_3d + e_elc

    # E(LT, LT)
    e_LT_total = get_layer_energy(q_LT, x_LT, y_LT, z_LT)
    
    # E(L-1 U L+1, L-1 U L+1)
    q_L1 = np.concatenate([q_L_minus1, q_L_plus1])
    z_L1 = np.concatenate([z_L_minus1 + shift, z_L_plus1 + shift])
    x_L1 = np.concatenate([xs[mask_L0_minus1], xs[mask_L0_plus1]])
    y_L1 = np.concatenate([ys[mask_L0_minus1], ys[mask_L0_plus1]])
    e_L1_total = get_layer_energy(q_L1, x_L1, y_L1, z_L1)

    # E(L0, L0)
    e_L0_total = get_layer_energy(qs, xs, ys, zs + shift)

    # Near-field contribution (Eq 4.14)
    e_near = 0.5 * (e_LT_total - e_L1_total + e_L0_total)

    # --- 5. Far-field Contribution ---
    # Restore original box for L2 calculation
    system.part.clear()
    system.box_l = original_box
    for i in range(N):
        system.part.add(pos=[xs[i], ys[i], zs[i]], q=qs[i])
    
    e_far = _calculate_L2_interaction(qs, xs, ys, zs, lx, ly, lz, lam, prefactor, p3m,
        delta_mid_bot, delta_mid_top, mask_L0_minus1, mask_L0_0, mask_L0_plus1
    )

    return {"e_near": e_near, "e_far": e_far, "total": e_near + e_far}

def _calculate_elc_correction(system, qs, xs, ys, zs, lx, ly, lz, gap_size, prefactor, p3m):
    """Calculate ELC correction term (Eq. 3.10 for non-neutral systems)."""
    
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    
    # Charge moments
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)
    
    # Constant term (Eq. 3.10, dipole correction for non-neutral)
    e_const = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)
    
    # Reciprocal space correction
    mesh_size = p3m.get_params()["mesh"]
    fx_max = mesh_size[0] / (2.0 * lx)
    fy_max = mesh_size[1] / (2.0 * ly)
    f_max = max(fx_max, fy_max)
    
    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))
    p, q = np.arange(-p_max, p_max + 1), np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx)**2 + (Q / ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    
    # Chi factors (Eq. 3.3, 3.6)
    def get_chi_terms(q_vec, z_vec):
        cx, sx = np.cos(arg_x[:, None] * xs), np.sin(arg_x[:, None] * xs)
        cy, sy = np.cos(arg_y[:, None] * ys), np.sin(arg_y[:, None] * ys)
        ex_p, ex_m = np.exp(arg_z[:, None] * z_vec), np.exp(-arg_z[:, None] * z_vec)
        
        def s(ez, c1, c2): return np.sum(q_vec * ez * c1 * c2, axis=1)
        
        return {
            "p": [s(ex_p, cx, cy), s(ex_p, sx, cy), s(ex_p, cx, sy), s(ex_p, sx, sy)],
            "m": [s(ex_m, cx, cy), s(ex_m, sx, cy), s(ex_m, cx, sy), s(ex_m, sx, sy)]
        }
    
    chi = get_chi_terms(qs, zs)
    
    # ELC reciprocal term (Eq. 3.5)
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    chi_val = sum(p_i * m_i for p_i, m_i in zip(chi["p"], chi["m"]))
    e_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi_val)
    
    return prefactor * (e_const + e_recip)


def _calculate_L2_interaction(
    qs, xs, ys, zs, lx, ly, lz, gap_size, prefactor, p3m,
    delta_mid_bot, delta_mid_top, mask_L0_minus1, mask_L0_0, mask_L0_plus1
):
    """
    Calculate interaction of L0 with L±2 using far formula (Section IV.A).
    This implements equations 4.4-4.12.
    """
    
    # Get reciprocal space mesh
    mesh_size = p3m.get_params()["mesh"]
    fx_max = mesh_size[0] / (2.0 * lx)
    fy_max = mesh_size[1] / (2.0 * ly)
    f_max = max(fx_max, fy_max)
    
    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))
    p, q = np.arange(-p_max, p_max + 1), np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx)**2 + (Q / ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    
    # Delta = Δb·Δt (product of dielectric contrasts)
    Delta = delta_mid_bot * delta_mid_top
    
    # L_p,q function (Eq. 4.4)
    def L_pq(z_arr, arg_z_val):
        """Compute geometric series sum for image charges."""
        exp_term = np.exp(-arg_z_val[:, None] * z_arr)
        return Delta * exp_term / (1.0 - Delta * np.exp(-4 * arg_z_val[:, None] * lz))
    
    # Helper to compute Chi factors
    def compute_chi_L2(q_subset, x_subset, y_subset, z_subset, L_factors):
        """Compute Chi factors for L±2 layers."""
        Tp_cc = np.cos(arg_x[:, None] * x_subset) * np.cos(arg_y[:, None] * y_subset)
        Tp_sc = np.sin(arg_x[:, None] * x_subset) * np.cos(arg_y[:, None] * y_subset)
        Tp_cs = np.cos(arg_x[:, None] * x_subset) * np.sin(arg_y[:, None] * y_subset)
        Tp_ss = np.sin(arg_x[:, None] * x_subset) * np.sin(arg_y[:, None] * y_subset)
        
        chi_cc = np.sum(q_subset * L_factors * Tp_cc, axis=1)
        chi_sc = np.sum(q_subset * L_factors * Tp_sc, axis=1)
        chi_cs = np.sum(q_subset * L_factors * Tp_cs, axis=1)
        chi_ss = np.sum(q_subset * L_factors * Tp_ss, axis=1)
        
        return chi_cc, chi_sc, chi_cs, chi_ss
    
    # --- L-2 contributions (Eq. 4.6-4.9) ---
    # From L0,-1 (near bottom): both series 2.3 and 2.4, excluding first element of 2.3
    # From L0,0 and L0,+1: both series complete
    
    chi_L0_cc = np.sum(qs * np.cos(arg_x[:, None] * xs) * np.cos(arg_y[:, None] * ys), axis=1)
    chi_L0_sc = np.sum(qs * np.sin(arg_x[:, None] * xs) * np.cos(arg_y[:, None] * ys), axis=1)
    chi_L0_cs = np.sum(qs * np.cos(arg_x[:, None] * xs) * np.sin(arg_y[:, None] * ys), axis=1)
    chi_L0_ss = np.sum(qs * np.sin(arg_x[:, None] * xs) * np.sin(arg_y[:, None] * ys), axis=1)
    
    # L-2 from bottom images
    L_minus2_cc = np.zeros_like(arg_z)
    L_minus2_sc = np.zeros_like(arg_z)
    L_minus2_cs = np.zeros_like(arg_z)
    L_minus2_ss = np.zeros_like(arg_z)
    
    if np.any(mask_L0_minus1):
        q_m1 = delta_mid_bot * qs[mask_L0_minus1]
        x_m1, y_m1, z_m1 = xs[mask_L0_minus1], ys[mask_L0_minus1], zs[mask_L0_minus1]
        L_fac = L_pq(2*lz + z_m1, arg_z) + L_pq(2*lz - z_m1, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_m1, x_m1, y_m1, z_m1, L_fac)
        L_minus2_cc += cc
        L_minus2_sc += sc
        L_minus2_cs += cs
        L_minus2_ss += ss
    
    if np.any(mask_L0_0 | mask_L0_plus1):
        mask_mid_top = mask_L0_0 | mask_L0_plus1
        q_mt = delta_mid_bot * qs[mask_mid_top]
        x_mt, y_mt, z_mt = xs[mask_mid_top], ys[mask_mid_top], zs[mask_mid_top]
        L_fac = L_pq(z_mt, arg_z) + L_pq(2*lz - z_mt, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_mt, x_mt, y_mt, z_mt, L_fac)
        L_minus2_cc += cc
        L_minus2_sc += sc
        L_minus2_cs += cs
        L_minus2_ss += ss
    
    # L+2 from top images (Eq. 4.10-4.12)
    L_plus2_cc = np.zeros_like(arg_z)
    L_plus2_sc = np.zeros_like(arg_z)
    L_plus2_cs = np.zeros_like(arg_z)
    L_plus2_ss = np.zeros_like(arg_z)
    
    if np.any(mask_L0_plus1):
        q_p1 = delta_mid_top * qs[mask_L0_plus1]
        x_p1, y_p1, z_p1 = xs[mask_L0_plus1], ys[mask_L0_plus1], zs[mask_L0_plus1]
        L_fac = L_pq(4*lz - z_p1, arg_z) + L_pq(2*lz + z_p1, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_p1, x_p1, y_p1, z_p1, L_fac)
        L_plus2_cc += cc
        L_plus2_sc += sc
        L_plus2_cs += cs
        L_plus2_ss += ss
    
    if np.any(mask_L0_0 | mask_L0_minus1):
        mask_mid_bot = mask_L0_0 | mask_L0_minus1
        q_mb = delta_mid_top * qs[mask_mid_bot]
        x_mb, y_mb, z_mb = xs[mask_mid_bot], ys[mask_mid_bot], zs[mask_mid_bot]
        L_fac = L_pq(2*lz - z_mb, arg_z) + L_pq(2*lz + z_mb, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_mb, x_mb, y_mb, z_mb, L_fac)
        L_plus2_cc += cc
        L_plus2_sc += sc
        L_plus2_cs += cs
        L_plus2_ss += ss
    
    # Combine using far formula (Eq. 3.4 structure)
    chi_prod = (L_minus2_cc * chi_L0_cc + L_minus2_sc * chi_L0_sc + 
                L_minus2_cs * chi_L0_cs + L_minus2_ss * chi_L0_ss +
                L_plus2_cc * chi_L0_cc + L_plus2_sc * chi_L0_sc +
                L_plus2_cs * chi_L0_cs + L_plus2_ss * chi_L0_ss)
    
    e_L2_recip = np.sum((1.0 / (lx * ly * f)) * chi_prod)
    
    # Constant/dipole correction terms (similar to Eq. 3.9)
    # For L±2 interaction, we need dipole corrections
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    
    # Simplified dipole term for image interaction
    e_L2_const = 0.0  # This would need proper dipole moment calculation for L±2
    
    return prefactor * (e_L2_const + e_L2_recip)


def get_elcic_energy(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top):
    """Calculate total ELCIC energy."""
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]