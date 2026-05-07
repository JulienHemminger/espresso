import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Calculate ELCIC energy contributions with full subdivision of charge layers.

    Parameters:
    -----------
    system : espressomd.System
        The simulation system
    gap_size : float
        Gap size (λ) for layer separation
    pw_error : float
        P3M accuracy target
    prefactor : float
        Electrostatic prefactor
    delta_mid_bot : float
        Dielectric contrast factor for bottom interface: (εm - εb)/(εm + εb)
    delta_mid_top : float
        Dielectric contrast factor for top interface: (εm - εt)/(εm + εt)

    Returns:
    --------
    dict : Energy contributions broken down by layer
    """

    # --- Setup P3M for 3D periodic system ---
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )

    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    N = len(qs)

    # --- Subdivide L0 based on distance to interfaces (Eq. 4.3) ---
    # L0,-1: charges with 0 ≤ z ≤ λ (near bottom)
    # L0,0:  charges with λ < z < lz - λ (middle)
    # L0,+1: charges with lz - λ ≤ z ≤ lz (near top)

    mask_L0_minus1 = (zs >= 0) & (zs <= gap_size)
    mask_L0_0 = (zs > gap_size) & (zs < lz - gap_size)
    mask_L0_plus1 = (zs >= lz - gap_size) & (zs <= lz)

    # --- Generate image charges for L±1 (Section IV.B) ---
    # L+1: first-generation images from L0,+1 (near top interface)
    # L-1: first-generation images from L0,-1 (near bottom interface)

    # Image charges for L+1 (top interface, first generation only - Eq. 2.5, first term)
    q_L_plus1 = delta_mid_top * qs[mask_L0_plus1]
    z_L_plus1 = 2 * lz - zs[mask_L0_plus1]
    x_L_plus1 = xs[mask_L0_plus1]
    y_L_plus1 = ys[mask_L0_plus1]

    # Image charges for L-1 (bottom interface, first generation only - Eq. 2.3, first term)
    q_L_minus1 = delta_mid_bot * qs[mask_L0_minus1]
    z_L_minus1 = -zs[mask_L0_minus1]
    x_L_minus1 = xs[mask_L0_minus1]
    y_L_minus1 = ys[mask_L0_minus1]

    # --- Create LT = L-1 ∪ L0 ∪ L+1 in expanded box (Section IV.B, Eq. 4.13) ---
    # We need an expanded box with gap = 3λ: Lz = lz + 3λ
    Lz = lz + 3 * gap_size

    # Combine all charges and positions for LT
    q_LT = np.concatenate([q_L_minus1, qs, q_L_plus1])
    x_LT = np.concatenate([x_L_minus1, xs, x_L_plus1])
    y_LT = np.concatenate([y_L_minus1, ys, y_L_plus1])
    z_LT = np.concatenate([z_L_minus1 + gap_size, zs + gap_size, z_L_plus1 + gap_size])

    # --- Calculate E(LT, LT) using P3M + ELC (Eq. 4.14) ---
    # Store original particles
    original_parts = [(p.pos.copy(), p.q) for p in system.part.all()]

    # Clear and setup expanded system
    system.part.clear()
    system.box_l = [lx, ly, Lz]

    # Add LT particles
    for i in range(len(q_LT)):
        system.part.add(pos=[x_LT[i], y_LT[i], z_LT[i]], q=q_LT[i])

    # Apply P3M + ELC for LT
    system.electrostatics.solver = p3m
    e_LT_3d = system.analysis.energy()["total"]

    # Calculate ELC correction for LT (non-neutral case, Eq. 3.10)
    e_LT_elc = _calculate_elc_correction(
        system, q_LT, x_LT, y_LT, z_LT, lx, ly, Lz, gap_size, prefactor, p3m
    )
    e_LT_total = e_LT_3d + e_LT_elc

    # --- Calculate E(L±1, L±1) using P3M + ELC ---
    # L±1 only contains image charges
    q_L1 = np.concatenate([q_L_minus1, q_L_plus1])
    x_L1 = np.concatenate([x_L_minus1, x_L_plus1])
    y_L1 = np.concatenate([y_L_minus1, y_L_plus1])
    z_L1 = np.concatenate([z_L_minus1 + gap_size, z_L_plus1 + gap_size])

    if len(q_L1) > 0:
        system.part.clear()
        for i in range(len(q_L1)):
            system.part.add(pos=[x_L1[i], y_L1[i], z_L1[i]], q=q_L1[i])

        e_L1_3d = system.analysis.energy()["total"]
        e_L1_elc = _calculate_elc_correction(
            system, q_L1, x_L1, y_L1, z_L1, lx, ly, Lz, gap_size, prefactor, p3m
        )
        e_L1_total = e_L1_3d + e_L1_elc
    else:
        e_L1_total = e_L1_3d = e_L1_elc = 0.0

    # --- Calculate E(L0, L0) using P3M + ELC ---
    system.part.clear()
    system.box_l = [lx, ly, Lz]

    for i in range(N):
        system.part.add(pos=[xs[i], ys[i], zs[i] + gap_size], q=qs[i])

    e_L0_3d = system.analysis.energy()["total"]
    e_L0_elc = _calculate_elc_correction(
        system, qs, xs, ys, zs + gap_size, lx, ly, Lz, gap_size, prefactor, p3m
    )
    e_L0_total = e_L0_3d + e_L0_elc

    # --- E(L0, LT) from Eq. 4.14 ---
    e_L0_LT = 0.5 * (e_LT_total - e_L1_total + e_L0_total)

    # --- Calculate E(L0, L±2) using far formula (Section IV.A) ---
    # Restore original box
    system.part.clear()
    system.box_l = [lx, ly, lz]
    for pos, q in original_parts:
        system.part.add(pos=pos, q=q)

    e_L2_total = _calculate_L2_interaction(
        qs, xs, ys, zs, lx, ly, lz, gap_size, prefactor, p3m,
        delta_mid_bot, delta_mid_top, mask_L0_minus1, mask_L0_0, mask_L0_plus1
    )

    # --- Total energy ---
    e_near = e_L0_LT    # "Near" includes L0 with L0 and L±1
    e_far = e_L2_total   # "Far" includes L0 with L±2

    return {
        "e_near": float(e_near),
        "e_far": float(e_far),
        "l0": {"e_3d": float(e_L0_3d), "e_elc": float(e_L0_elc), "total": float(e_L0_total)},
        "pm1": {"e_3d": float(e_L1_3d), "e_elc": float(e_L1_elc), "total": float(e_L1_total)},
        "lt": {"e_3d": float(e_LT_3d), "e_elc": float(e_LT_elc), "total": float(e_LT_total)},
        "e_far_detail": {"L2": float(e_L2_total)}
    }


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
    params = p3m.get_params()
    mesh_size = params["mesh"]
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

        def s(ez, c1, c2):
            return np.sum(q_vec * ez * c1 * c2, axis=1)

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
    params = p3m.get_params()
    mesh_size = params["mesh"]
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
    chi_L0_cc = np.sum(qs * np.cos(arg_x[:, None] * xs) * np.cos(arg_y[:, None] * ys), axis=1)
    chi_L0_sc = np.sum(qs * np.sin(arg_x[:, None] * xs) * np.cos(arg_y[:, None] * ys), axis=1)
    chi_L0_cs = np.sum(qs * np.cos(arg_x[:, None] * xs) * np.sin(arg_y[:, None] * ys), axis=1)
    chi_L0_ss = np.sum(qs * np.sin(arg_x[:, None] * xs) * np.sin(arg_y[:, None] * ys), axis=1)

    L_minus2_cc = np.zeros_like(arg_z)
    L_minus2_sc = np.zeros_like(arg_z)
    L_minus2_cs = np.zeros_like(arg_z)
    L_minus2_ss = np.zeros_like(arg_z)

    if np.any(mask_L0_minus1):
        q_m1 = delta_mid_bot * qs[mask_L0_minus1]
        x_m1, y_m1, z_m1 = xs[mask_L0_minus1], ys[mask_L0_minus1], zs[mask_L0_minus1]
        L_fac = L_pq(2 * lz + z_m1, arg_z) + L_pq(2 * lz - z_m1, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_m1, x_m1, y_m1, z_m1, L_fac)
        L_minus2_cc += cc
        L_minus2_sc += sc
        L_minus2_cs += cs
        L_minus2_ss += ss

    if np.any(mask_L0_0 | mask_L0_plus1):
        mask_mid_top = mask_L0_0 | mask_L0_plus1
        q_mt = delta_mid_bot * qs[mask_mid_top]
        x_mt, y_mt, z_mt = xs[mask_mid_top], ys[mask_mid_top], zs[mask_mid_top]
        L_fac = L_pq(z_mt, arg_z) + L_pq(2 * lz - z_mt, arg_z)
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
        L_fac = L_pq(4 * lz - z_p1, arg_z) + L_pq(2 * lz + z_p1, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_p1, x_p1, y_p1, z_p1, L_fac)
        L_plus2_cc += cc
        L_plus2_sc += sc
        L_plus2_cs += cs
        L_plus2_ss += ss

    if np.any(mask_L0_0 | mask_L0_minus1):
        mask_mid_bot = mask_L0_0 | mask_L0_minus1
        q_mb = delta_mid_top * qs[mask_mid_bot]
        x_mb, y_mb, z_mb = xs[mask_mid_bot], ys[mask_mid_bot], zs[mask_mid_bot]
        L_fac = L_pq(2 * lz - z_mb, arg_z) + L_pq(2 * lz + z_mb, arg_z)
        cc, sc, cs, ss = compute_chi_L2(q_mb, x_mb, y_mb, z_mb, L_fac)
        L_plus2_cc += cc
        L_plus2_sc += sc
        L_plus2_cs += cs
        L_plus2_ss += ss

    # Combine using far formula
    chi_prod = (
        (L_minus2_cc + L_plus2_cc) * chi_L0_cc +
        (L_minus2_sc + L_plus2_sc) * chi_L0_sc +
        (L_minus2_cs + L_plus2_cs) * chi_L0_cs +
        (L_minus2_ss + L_plus2_ss) * chi_L0_ss
    )

    e_L2_recip = np.sum((1.0 / (lx * ly * f)) * chi_prod)
    e_L2_const = 0.0  # Placeholder for dipole corrections

    return prefactor * (e_L2_const + e_L2_recip)


def get_elcic_energy(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top):
    """Calculate total ELCIC energy."""
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]