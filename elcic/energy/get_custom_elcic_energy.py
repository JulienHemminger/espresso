import espressomd
import espressomd.electrostatics
import numpy as np


def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Equation 3.3: Product decomposition for the ELC reciprocal sum."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T

    # Use real/imaginary parts to represent sin/cos product decomposition
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])

    cx = np.cos(arg_x * xs[:, None])
    sx = np.sin(arg_x * xs[:, None])
    cy = np.cos(arg_y * ys[:, None])
    sy = np.sin(arg_y * ys[:, None])

    # Eq 3.3: Summing q * exp * trig_x * trig_y
    return [
        np.sum(qs[:, None] * ez * c1 * c2, axis=0)
        for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]
    ]


def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """Evaluates Equations 4.6 through 4.12 for far image interactions."""
    lx, ly, lz = box
    delta = db * dt
    # Frequency cutoff based on gap size and required precision
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

    # Eq 3.3 components for the real charges
    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        """Eq 4.4: Geometric series sum for multiple reflections."""
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    # L-2 (Bottom far field)
    m_bot = ps[:, 2] <= gap_size
    m_near_top = ~m_bot

    # Eq 4.6: Combining reflections for chi_m2
    chi_m2_p = [np.zeros_like(f) for _ in range(4)]
    if np.any(m_bot):
        chi_local = _get_chi_components(fx, fy, f, ps[m_bot], qs[m_bot], sign=0)
        # Term 1: q * (db * delta * L(2lz + z) + delta * L(2lz - z))
        t1 = l_pq_sum(2 * lz + ps[m_bot, 2, None], db * delta) + l_pq_sum(
            2 * lz - ps[m_bot, 2, None], delta
        )
        term_sum = np.sum(t1, axis=0)
        chi_m2_p = [chi_m2_p[i] + chi_local[i] * term_sum for i in range(4)]

    if np.any(m_near_top):
        chi_local = _get_chi_components(
            fx, fy, f, ps[m_near_top], qs[m_near_top], sign=0
        )
        # Term 2: q * (db * L(z) + delta * L(2lz - z))
        t2 = l_pq_sum(ps[m_near_top, 2, None], db) + l_pq_sum(
            2 * lz - ps[m_near_top, 2, None], delta
        )
        term_sum = np.sum(t2, axis=0)
        chi_m2_p = [chi_m2_p[i] + chi_local[i] * term_sum for i in range(4)]

    # L+2 (Top far field)
    m_top = ps[:, 2] > (lz - gap_size)
    m_near_bot = ~m_top

    chi_p2_m = [np.zeros_like(f) for _ in range(4)]
    if np.any(m_top):
        chi_local = _get_chi_components(fx, fy, f, ps[m_top], qs[m_top], sign=0)
        # Eq 4.10 logic
        t1 = l_pq_sum(4 * lz - ps[m_top, 2, None], dt * delta) + l_pq_sum(
            2 * lz + ps[m_top, 2, None], delta
        )
        term_sum = np.sum(t1, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    if np.any(m_near_bot):
        chi_local = _get_chi_components(
            fx, fy, f, ps[m_near_bot], qs[m_near_bot], sign=0
        )
        t2 = l_pq_sum(2 * lz - ps[m_near_bot, 2, None], dt) + l_pq_sum(
            2 * lz + ps[m_near_bot, 2, None], delta
        )
        term_sum = np.sum(t2, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    # Energy summation Eq 3.4
    pref = 0.5 / (lx * ly)
    e_far = 0.0
    for c0_m, cm2_p in zip(chi0_m, chi_m2_p):
        e_far += pref * np.sum((1.0 / f) * c0_m * cm2_p)
    for c0_p, cp2_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * c0_p * cp2_m)

    return e_far


def _get_non_neutral_correction(box, zs, qs):
    """Equation 3.10: Corrected dipole and slab term for ELC."""
    lx, ly, Lz = box
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)

    # ELC specific dipole correction (Eq 3.10)
    # fac = 2pi / Vol
    fac = 2.0 * np.pi / (lx * ly * Lz)

    # Correction term: subtracts 3D dipole, adds 2D dipole component
    # For neutral systems (xi0=0), this simplifies to the xi1**2 term.
    term = xi1**2 - xi0 * xi2 - (Lz**2 / 12.0) * xi0**2
    return fac * term


def _get_config_energy(
    system, p_set, q_set, prefactor, accuracy, gap_size, physical_lz
):
    """Computes P3M + ELC dipole correction (Eq 3.10)."""
    lx, ly, _ = system.box_l
    extended_lz = physical_lz + gap_size

    system.part.clear()
    system.box_l = [lx, ly, extended_lz]
    system.part.add(pos=p_set, q=q_set)

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=accuracy,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]

    # Apply correction based on the non-neutral formulation in Eq 3.10
    e_corr = prefactor * _get_non_neutral_correction(system.box_l, p_set[:, 2], q_set)

    system.electrostatics.clear()
    return e_3d + e_corr


def get_elcic_energy(system, params: dict):
    box = np.array(system.box_l)
    lz = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]

    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    # 1. Identify Near-Images (Section IV)
    m_bot = ps_orig[:, 2] <= gap
    m_top = ps_orig[:, 2] > (lz - gap)

    ps_m1 = ps_orig[m_bot].copy()
    ps_m1[:, 2] *= -1  # Reflection at z=0

    ps_p1 = ps_orig[m_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]  # Reflection at z=lz

    qs_m1 = qs_orig[m_bot] * db
    qs_p1 = qs_orig[m_top] * dt

    # 2. Near-Field Energy (Eq 4.14)
    # Φ(L0, LT) = 0.5 * (Φ(LT, LT) - Φ(L±1, L±1) + Φ(L0, L0))
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps, gap, lz)

    has_images = len(qs_m1) > 0 or len(qs_p1) > 0
    if has_images:
        ps_img = np.vstack([p for p in [ps_m1, ps_p1] if len(p) > 0])
        qs_img = np.concatenate([q for q in [qs_m1, qs_p1] if len(q) > 0])

        e_pm1 = _get_config_energy(system, ps_img, qs_img, pref, eps, gap, lz)

        ps_total = np.vstack([ps_orig, ps_img])
        qs_total = np.concatenate([qs_orig, qs_img])
        e_lt = _get_config_energy(system, ps_total, qs_total, pref, eps, gap, lz)

        e_near = 0.5 * (e_lt - e_pm1 + e_l0)
    else:
        e_near = e_l0

    # 3. Far-Field Energy (Eq 4.6 - 4.12)
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    # Cleanup
    system.part.clear()
    system.box_l = box
    system.part.add(pos=ps_orig, q=qs_orig)

    return e_near + e_far
