import espressomd
import espressomd.electrostatics
import numpy as np


def _get_e_non_neutral_corr(lx, ly, lz, zs, qs):
    """Equation 3.10: Dipole and non-neutrality energy correction."""
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    fac = 2.0 * np.pi / (lx * ly * lz)
    return fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)


def _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1):
    """Equation 3.3: Components for product decomposition."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    ez = np.exp(sign * arg_z * zs[:, None]) if sign != 0 else 1.0
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    return [
        np.sum(qs[:, None] * ez * c1 * c2, axis=0)
        for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]
    ]


def _get_e_far_field(lx, ly, lz, gap_size, pw_error, qs, xs, ys, zs, db, dt):
    """Calculates the analytic Far-Field energy (L-2 and L+2 image sets)."""
    delta = db * dt
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

    m_bot = zs <= gap_size
    m_top = zs > (lz - gap_size)

    chi0_p = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=1)
    chi0_m = _get_chi(fx, fy, f, xs, ys, zs, qs, sign=-1)

    def get_L_pq(z_dist):
        return (delta * np.exp(-2.0 * np.pi * f * z_dist)) / (
            1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        )

    chi_m2_p = [np.zeros_like(f) for _ in range(4)]
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]

    chi_bot = _get_chi(fx, fy, f, xs[m_bot], ys[m_bot], zs[m_bot], qs[m_bot], sign=0)
    chi_rest = _get_chi(
        fx, fy, f, xs[~m_bot], ys[~m_bot], zs[~m_bot], qs[~m_bot], sign=0
    )
    chi_top = _get_chi(fx, fy, f, xs[m_top], ys[m_top], zs[m_top], qs[m_top], sign=0)
    chi_not_top = _get_chi(
        fx, fy, f, xs[~m_top], ys[~m_top], zs[~m_top], qs[~m_top], sign=0
    )

    for i in range(4):
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

    pref = -1.0 / (lx * ly)
    e_far = pref * np.sum((1.0 / f) * sum(chi0_m[i] * chi_m2_p[i] for i in range(4)))
    e_far += pref * np.sum((1.0 / f) * sum(chi0_p[i] * chi_p2_m[i] for i in range(4)))

    return e_far


def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Computes ELCIC energy and returns a detailed breakdown of all contributions.
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, ps = parts.q, parts.pos

    # 1. Near-Image Setup
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)

    qs_m1, ps_m1 = qs[m_bot] * delta_mid_bot, ps[m_bot].copy()
    ps_m1[:, 2] = -ps_m1[:, 2]
    qs_p1, ps_p1 = qs[m_top] * delta_mid_top, ps[m_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]

    def get_set_details(q_set, p_set):
        system.part.clear()
        system.part.add(pos=p_set, q=q_set)
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False
        )
        system.electrostatics.solver = p3m
        system.integrator.run(0)
        e_3d = system.analysis.energy()["total"]
        e_corr = prefactor * _get_e_non_neutral_corr(lx, ly, lz, p_set[:, 2], q_set)
        return {"e_3d": e_3d, "e_corr": e_corr, "total": e_3d + e_corr}

    # Compute the three sets required for Near energy decomposition (Eq. 4.14)
    res_l0 = get_set_details(qs, ps)
    res_pm1 = get_set_details(
        np.concatenate([qs_m1, qs_p1]), np.concatenate([ps_m1, ps_p1])
    )
    res_lt = get_set_details(
        np.concatenate([qs, qs_m1, qs_p1]), np.concatenate([ps, ps_m1, ps_p1])
    )

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

    contribs = {
        "l0": res_l0,  # Real charges set
        "pm1": res_pm1,  # Near images only (L-1 + L+1)
        "lt": res_lt,  # Combined near set (L0 + L-1 + L+1)
        "e_far": e_far,  # Analytic far-field contribution
        "e_near": 0.5 * (res_lt["total"] - res_pm1["total"] + res_l0["total"]),
    }
    return contribs


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Backward compatible method that returns the total ELCIC energy.
    """
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]
