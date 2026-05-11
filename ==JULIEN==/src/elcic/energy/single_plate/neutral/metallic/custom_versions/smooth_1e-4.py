import espressomd
import espressomd.electrostatics
import numpy as np


def _get_e_non_neutral_corr(lx, ly, lz, zs, qs):
    """
    Equation 3.10 & 3.11: Dipole and non-neutrality energy correction.
    Handles the parabolic background term required for non-neutral slabs.
    """
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    fac = 2.0 * np.pi / (lx * ly * lz)
    return fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)


def _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1):
    """
    Equation 3.3: Product decomposition for the ELC reciprocal sum.
    Returns the four trigonometric components (cos*cos, sin*cos, cos*sin, sin*sin).
    """
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    # If sign=0, we only want the lateral part (for image charge source terms)
    ez = np.exp(sign * arg_z * zs[:, None]) if sign != 0 else 1.0

    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    return [
        np.sum(qs[:, None] * ez * c1 * c2, axis=0)
        for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]
    ]


def _get_e_far_field(lx, ly, lz, gap_size, pw_error, qs, xs, ys, zs, db, dt):
    """
    Evaluates Equations 4.6 through 4.13.
    Calculates the interaction energy between the real charges and the infinite
    sets of far image charges (L-2 and L+2) using the analytic L_pq sum.
    """
    delta = db * dt
    # Convergence criteria for reciprocal sum
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    inner_mask = f <= f_max
    fx, fy, f = fx[inner_mask], fy[inner_mask], f[inner_mask]

    # Define geometric subsets L0_bot, L0_top, L0_mid (Eq. 4.3)
    m_bot = zs <= gap_size
    m_top = zs > (lz - gap_size)

    # Base chi factors for the real charges (Eq. 3.4)
    chi0_p = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1)
    chi0_m = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=-1)

    def get_L_pq(z_dist):
        # Equation 4.4: Analytical sum of the geometric series of images
        return (delta * np.exp(-2.0 * np.pi * f * z_dist)) / (
            1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        )

    # Initialize Far-Image factors
    chi_m2_p = [np.zeros_like(f) for _ in range(4)]
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]

    # Interaction factors for L-2 (bottom far images)
    # This combines Equations 4.6, 4.7, and 4.8
    chi_bot = _get_chi(fx, fy, f, xs[m_bot], ys[m_bot], zs[m_bot], qs[m_bot], sign=0)
    chi_rest = _get_chi(
        fx, fy, f, xs[~m_bot], ys[~m_bot], zs[~m_bot], qs[~m_bot], sign=0
    )

    # Interaction factors for L+2 (top far images)
    # This combines Equations 4.10, 4.11, and 4.12
    chi_top = _get_chi(fx, fy, f, xs[m_top], ys[m_top], zs[m_top], qs[m_top], sign=0)
    chi_not_top = _get_chi(
        fx, fy, f, xs[~m_top], ys[~m_top], zs[~m_top], qs[~m_top], sign=0
    )

    for i in range(4):
        # Contribution to chi_m2^+ (Eq. 4.6)
        chi_m2_p[i] = chi_bot[i] * np.sum(
            db * delta * get_L_pq(2 * lz + zs[m_bot, None])
            + delta * get_L_pq(2 * lz - zs[m_bot, None]),
            axis=0,
        )
        chi_m2_p[i] += chi_rest[i] * np.sum(
            db * get_L_pq(zs[~m_bot, None])
            + delta * get_L_pq(2 * lz - zs[~m_bot, None]),
            axis=0,
        )

        # Contribution to chi_p2^- (Eq. 4.10)
        chi_p2_m[i] = chi_top[i] * np.sum(
            dt * delta * get_L_pq(2 * lz + (lz - zs[m_top, None]))
            + delta * get_L_pq(2 * lz - (lz - zs[m_top, None])),
            axis=0,
        )
        chi_p2_m[i] += chi_not_top[i] * np.sum(
            dt * get_L_pq(lz - zs[~m_top, None])
            + delta * get_L_pq(2 * lz - (lz - zs[~m_top, None])),
            axis=0,
        )

    # Total Far Field Energy (Eq. 4.13)
    pref = -1.0 / (lx * ly)
    e_far = pref * np.sum((1.0 / f) * sum(chi0_m[i] * chi_m2_p[i] for i in range(4)))
    e_far += pref * np.sum((1.0 / f) * sum(chi0_p[i] * chi_p2_m[i] for i in range(4)))

    return e_far


def get_elcic_energy_old(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Main ELCIC energy calculation.
    Combines the near-field (P3M + ELC) and the far-field image corrections.
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, ps = parts.q, parts.pos

    # 1. Separate Near-Image sources (L-1 and L+1)
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)

    qs_m1, ps_m1 = qs[m_bot] * delta_mid_bot, ps[m_bot].copy()
    ps_m1[:, 2] = -ps_m1[:, 2]

    qs_p1, ps_p1 = qs[m_top] * delta_mid_top, ps[m_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]

    def get_set_energy(q_set, p_set):
        # Helper to compute 3D periodic energy and apply dipole/neutrality correction
        system.part.clear()
        system.part.add(pos=p_set, q=q_set)
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False
        )
        system.electrostatics.solver = p3m
        e_3d = system.analysis.energy()["total"]
        e_corr = prefactor * _get_e_non_neutral_corr(lx, ly, lz, p_set[:, 2], q_set)
        return e_3d + e_corr

    # 2. Linear combination of sets to isolate near interactions (Eq. 4.14)
    # Energy of real charges (L0)
    e_l0 = get_set_energy(qs, ps)
    # Energy of near images only (L-1 U L+1)
    e_pm1 = get_set_energy(
        np.concatenate([qs_m1, qs_p1]), np.concatenate([ps_m1, ps_p1])
    )
    # Energy of full near set (L0 U L-1 U L+1)
    e_lt = get_set_energy(
        np.concatenate([qs, qs_m1, qs_p1]), np.concatenate([ps, ps_m1, ps_p1])
    )

    # 3. Final summation
    e_near = 0.5 * (e_lt - e_pm1 + e_l0)
    e_far = prefactor * _get_e_far_field(
        lx,
        ly,
        lz,
        gap_size,
        pw_error,
        qs,
        ps[:, 0],
        ps[:, 1],
        ps[:, 2],
        delta_mid_bot,
        delta_mid_top,
    )

    # Restore original system state
    system.part.clear()
    system.part.add(pos=ps, q=qs)

    return e_near + e_far


def get_elcic_energy(system, params: dict):
    """Wrapper that extracts params and calls the main calculation."""
    return get_elcic_energy_old(
        system, 
        gap_size=params["gap_size"], 
        pw_error=params["pw_error"], 
        prefactor=params.get("prefactor", 1.0), 
        delta_mid_bot=params["delta_mid_bot"], 
        delta_mid_top=params["delta_mid_top"]
    )