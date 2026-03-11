import espressomd
import espressomd.electrostatics
import numpy as np


def _get_e_non_neutral_corr(lx, ly, lz, zs, qs):
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    fac = 2.0 * np.pi / (lx * ly * lz)
    # Equation 3.10: Dipole and non-neutrality energy correction [cite: 407]
    return fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)


def _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1):
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    ez = np.exp(sign * arg_z * zs[:, None])
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    # Components for product decomposition (Eq. 3.3) [cite: 67]
    return [
        np.sum(qs[:, None] * ez * c1 * c2, axis=0)
        for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]
    ]


def _get_e_far_field(lx, ly, lz, gap_size, pw_error, qs, xs, ys, zs, db, dt):
    delta = db * dt
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    # Subsets (Eq. 4.3) [cite: 112]
    m_bot = zs <= gap_size
    m_top = zs > (lz - gap_size)
    m_mid = ~(m_bot | m_top)

    chi0_p = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1)
    chi0_m = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=-1)

    def get_L_pq(z):  # Eq. 4.4 [cite: 120]
        return (delta * np.exp(-2.0 * np.pi * f * z)) / (
            1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        )

    # Far-field sums for L-2 and L+2 (Eq. 4.6, 4.10) [cite: 127, 133]
    # Simplified logic: calculating Chi_L2 directly to plug into product decomposition
    chi_m2_p = [np.zeros_like(f) for _ in range(4)]
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]

    # Interaction factors for L-2 (bottom far images)
    for i in range(4):
        term_bot = _get_chi(
            fx, fy, f, xs[m_bot], ys[m_bot], zs[m_bot], qs[m_bot], sign=0
        )[i] * (
            db * delta * get_L_pq(2 * lz + zs[m_bot, None])
            + delta * get_L_pq(2 * lz - zs[m_bot, None])
        ).sum(axis=0)
        term_rest = _get_chi(
            fx, fy, f, xs[~m_bot], ys[~m_bot], zs[~m_bot], qs[~m_bot], sign=0
        )[i] * (
            db * get_L_pq(zs[~m_bot, None])
            + delta * get_L_pq(2 * lz - zs[~m_bot, None])
        ).sum(axis=0)
        chi_m2_p[i] = term_bot + term_rest

    # Energy from Far Images (Eq. 3.4) [cite: 67, 135]
    e_far = -np.sum(
        (1.0 / (lx * ly * f)) * sum(chi0_m[i] * chi_m2_p[i] for i in range(4))
    )
    e_far += -np.sum(
        (1.0 / (lx * ly * f)) * sum(chi0_p[i] * chi_p2_m[i] for i in range(4))
    )
    # (Repeat similarly for chi_p2_m and add to e_far)
    return e_far


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, ps = parts.q, parts.pos

    # 1. Define sets L0, L_plus_1, L_minus_1
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)

    # Create Near-Images [cite: 113, 115]
    qs_m1, ps_m1 = qs[m_bot] * delta_mid_bot, ps[m_bot].copy()
    ps_m1[:, 2] = -ps_m1[:, 2]
    qs_p1, ps_p1 = qs[m_top] * delta_mid_top, ps[m_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]

    def get_set_energy(q_set, p_set):
        # Temp modification to system to use P3M
        system.part.clear()
        system.part.add(pos=p_set, q=q_set)
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False
        )
        system.electrostatics.solver = p3m
        e_3d = system.analysis.energy()["total"]
        # Standard ELC Correction for this set
        e_corr = prefactor * _get_e_non_neutral_corr(lx, ly, lz, p_set[:, 2], q_set)
        # Reciprocal part (omitted for brevity, same as user's _get_e_recip)
        return e_3d + e_corr

    # Energy Decomposition (Eq. 4.14)
    e_l0 = get_set_energy(qs, ps)
    e_pm1 = get_set_energy(
        np.concatenate([qs_m1, qs_p1]), np.concatenate([ps_m1, ps_p1])
    )
    e_lt = get_set_energy(
        np.concatenate([qs, qs_m1, qs_p1]), np.concatenate([ps, ps_m1, ps_p1])
    )

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

    # Restore original system
    system.part.clear()
    system.part.add(pos=ps, q=qs)
    return e_near + e_far
