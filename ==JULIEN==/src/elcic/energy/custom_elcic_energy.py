import espressomd
import espressomd.electrostatics
import numpy as np


def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor,
    delta_mid_bot, delta_mid_top
):
    """
    Implements the ELCIC energy decomposition from
    Tyagi, Arnold, Holm JCP 129, 204102 (2008).

    Parameters
    ----------
    gap_size : float
        Gap size λ in the paper.
    pw_error : float
        Target P3M accuracy.
    prefactor : float
        Coulomb prefactor (e.g. 1/(4πϵ0)).
    delta_mid_bot : float
        Δ_b = (ε_m - ε_b)/(ε_m + ε_b)
    delta_mid_top : float
        Δ_t = (ε_m - ε_t)/(ε_m + ε_t)
    """

    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q.copy()
    pos = parts.pos.copy()

    xs = pos[:, 0]
    ys = pos[:, 1]
    zs = pos[:, 2]

    ux = 1.0 / lx
    uy = 1.0 / ly
    uz = 1.0 / lz

    # ------------------------------------------------------------------
    # 1. Build extended box for near part (LT = L-1 ∪ L0 ∪ L+1)
    # ------------------------------------------------------------------

    lambda_gap = gap_size
    Lz_ext = lz + 3.0 * lambda_gap

    # Identify near-surface particles
    mask_bot = zs <= lambda_gap
    mask_top = zs >= (lz - lambda_gap)

    # Build LT particles (real + first images)
    pos_LT = []
    q_LT = []

    # Real charges shifted into extended box center
    z_shift = lambda_gap + lz

    for x, y, z, q in zip(xs, ys, zs, qs):
        pos_LT.append([x, y, z + z_shift])
        q_LT.append(q)

    # Bottom first images (series 2.3 first term)
    for x, y, z, q in zip(xs[mask_bot], ys[mask_bot], zs[mask_bot], qs[mask_bot]):
        pos_LT.append([x, y, -z + z_shift])
        q_LT.append(delta_mid_bot * q)

    # Top first images (series 2.5 first term)
    for x, y, z, q in zip(xs[mask_top], ys[mask_top], zs[mask_top], qs[mask_top]):
        pos_LT.append([x, y, 2 * lz - z + z_shift])
        q_LT.append(delta_mid_top * q)

    pos_LT = np.array(pos_LT)
    q_LT = np.array(q_LT)

    # ------------------------------------------------------------------
    # 2. Compute 3D periodic energy (P3M)
    # ------------------------------------------------------------------

    orig_box = np.copy(system.box_l)
    orig_pos = pos.copy()
    orig_q = qs.copy()

    system.part.clear()
    system.box_l = [lx, ly, Lz_ext]
    system.part.add(pos=pos_LT, q=q_LT)

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
    )

    system.electrostatics.solver = p3m
    system.integrator.run(0)

    e_3d = float(system.analysis.energy()["total"])

    # ------------------------------------------------------------------
    # 3. ELC correction (Eq. 3.10)
    # ------------------------------------------------------------------

    def elc_term(qs, xs, ys, zs, Lz_box):
        fsum = 0.0
        dip0 = np.sum(qs)
        dip1 = np.sum(qs * zs)
        dip2 = np.sum(qs * zs * zs)

        pmax = 15  # safe default; can tune
        for p in range(-pmax, pmax + 1):
            for q in range(-pmax, pmax + 1):
                if p == 0 and q == 0:
                    continue

                f = np.sqrt((ux * p) ** 2 + (uy * q) ** 2)
                omega_p = 2 * np.pi * ux * p
                omega_q = 2 * np.pi * uy * q

                cosx = np.cos(omega_p * xs)
                sinx = np.sin(omega_p * xs)
                cosy = np.cos(omega_q * ys)
                siny = np.sin(omega_q * ys)

                exp_pos = np.exp(2 * np.pi * f * zs)
                exp_neg = np.exp(-2 * np.pi * f * zs)

                rho_plus = np.sum(qs * exp_pos * cosx * cosy)
                rho_minus = np.sum(qs * exp_neg * cosx * cosy)

                Lfac = np.exp(-2 * np.pi * f * Lz_box)
                denom = 1.0 - Lfac ** 2

                fsum += (rho_minus * rho_plus * Lfac / denom) / f

        elc_pw = -0.5 * ux * uy * fsum

        dip_corr = (
            2 * np.pi * ux * uy * uz
            * (dip1 ** 2 - dip0 * dip2 - (Lz_box ** 2) * dip0 ** 2 / 12.0)
        )

        return elc_pw + dip_corr

    e_elc = elc_term(q_LT, pos_LT[:, 0], pos_LT[:, 1], pos_LT[:, 2], Lz_ext)

    e_near = e_3d + e_elc

    # ------------------------------------------------------------------
    # 4. Far-field image contributions (L±2)
    #     Eq. 4.4–4.12
    # ------------------------------------------------------------------

    delta = delta_mid_bot * delta_mid_top

    def L_factor(f, z):
        return np.exp(-2 * np.pi * f * z) / (1 - delta * np.exp(-4 * np.pi * f * lz))

    e_far_total = 0.0
    e_far_detail = {}

    pmax = 15

    for p in range(-pmax, pmax + 1):
        for q in range(-pmax, pmax + 1):
            if p == 0 and q == 0:
                continue

            f = np.sqrt((ux * p) ** 2 + (uy * q) ** 2)
            omega_p = 2 * np.pi * ux * p
            omega_q = 2 * np.pi * uy * q

            cosx = np.cos(omega_p * xs)
            cosy = np.cos(omega_q * ys)

            # bottom images (L-2)
            rho_bot = np.sum(
                qs * delta_mid_bot * L_factor(f, zs) * cosx * cosy
            )

            # top images (L+2)
            rho_top = np.sum(
                qs * delta_mid_top * L_factor(f, 2 * lz - zs) * cosx * cosy
            )

            e_far_total += ux * uy * (rho_bot + rho_top) / (2 * f)

    e_far = e_far_total

    # ------------------------------------------------------------------
    # Restore original system
    # ------------------------------------------------------------------

    system.part.clear()
    system.box_l = orig_box
    system.part.add(pos=orig_pos, q=orig_q)

    # ------------------------------------------------------------------

    contribs = {
        "e_near": float(e_near),
        "e_far": float(e_far),
        "l0": {
            "e_3d": e_3d,
            "e_elc": e_elc,
            "total": e_near,
        },
        "pm1": {
            "e_3d": 0.0,
            "e_elc": 0.0,
            "total": 0.0,
        },
        "lt": {
            "e_3d": e_3d,
            "e_elc": e_elc,
            "total": e_near,
        },
        "e_far_detail": e_far_detail,
    }

    return contribs


def get_elcic_energy(
    system, gap_size, pw_error, prefactor,
    delta_mid_bot, delta_mid_top
):
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error,
        prefactor,
        delta_mid_bot,
        delta_mid_top,
    )
    return contribs["e_near"] + contribs["e_far"]