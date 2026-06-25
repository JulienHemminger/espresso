import numpy as np
from scipy.special import erfcx, erf, erfc


def get_ewald2d_elcic(params, k_max=10, n_real=10, tol=1e-8):
    pos = np.asarray(params["positions"], dtype=np.float64)
    q = np.asarray(params["charges"], dtype=np.float64)

    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_mid_bot, delta_mid_top = params["delta_mid_bot"], params["delta_mid_top"]

    area = lx * ly

    alpha = None
    if alpha is None:
        alpha = 5.0 / min(lx, ly)

    # Mirror charges for dielectric contrast at z = 0
    pos_mirror = pos.copy()
    pos_mirror[:, 2] = -pos_mirror[:, 2]
    q_mirror = delta_mid_bot * q

    def _h(k, z):
        u_plus = k / (2 * alpha) + alpha * z
        u_minus = k / (2 * alpha) - alpha * z
        prefac = np.exp(-(k**2) / (4 * alpha**2) - (alpha * z) ** 2)
        return prefac * (erfcx(u_plus) + erfcx(u_minus))

    def _real_space(pos_a, q_a, pos_b, q_b, exclude_self_n0):
        E = 0.0
        for nx in range(-n_real, n_real + 1):
            for ny in range(-n_real, n_real + 1):
                sx = nx * lx
                sy = ny * ly
                for i in range(len(q_a)):
                    for j in range(len(q_b)):
                        if exclude_self_n0 and i == j and nx == 0 and ny == 0:
                            continue
                        dx = pos_b[j, 0] - pos_a[i, 0] + sx
                        dy = pos_b[j, 1] - pos_a[i, 1] + sy
                        dz = pos_b[j, 2] - pos_a[i, 2]
                        r = np.sqrt(dx * dx + dy * dy + dz * dz)
                        E += q_a[i] * q_b[j] * erfc(alpha * r) / r
        return 0.5 * E

    def _recip_space(pos_a, q_a, pos_b, q_b):
        E = 0.0
        two_pi_over_Lx = 2 * np.pi / lx
        two_pi_over_Ly = 2 * np.pi / ly
        for mx in range(-k_max, k_max + 1):
            for my in range(-k_max, k_max + 1):
                if mx == 0 and my == 0:
                    continue
                kx = two_pi_over_Lx * mx
                ky = two_pi_over_Ly * my
                k = np.sqrt(kx * kx + ky * ky)
                for i in range(len(q_a)):
                    for j in range(len(q_b)):
                        drho_x = pos_b[j, 0] - pos_a[i, 0]
                        drho_y = pos_b[j, 1] - pos_a[i, 1]
                        z_ij = pos_b[j, 2] - pos_a[i, 2]
                        k_dot_rho = kx * drho_x + ky * drho_y
                        E += q_a[i] * q_b[j] * np.cos(k_dot_rho) * _h(k, z_ij) / k
        return 0.5 * (np.pi / area) * E

    def _k0_term(pos_a, q_a, pos_b, q_b):
        E = 0.0
        inv_a_sqrtpi = 1.0 / (alpha * np.sqrt(np.pi))
        for i in range(len(q_a)):
            for j in range(len(q_b)):
                z_ij = pos_b[j, 2] - pos_a[i, 2]
                abs_z = np.abs(z_ij)
                E += (
                    q_a[i]
                    * q_b[j]
                    * (
                        abs_z * erf(alpha * abs_z)
                        + np.exp(-((alpha * z_ij) ** 2)) * inv_a_sqrtpi
                    )
                )
        return -1.0 * (np.pi / area) * E

    def _self_energy(q):
        return -(alpha / np.sqrt(np.pi)) * np.sum(q**2)

    # --- real–real contribution (standard 2D Ewald with self-exclusion) ---
    E_real_real = (
        _real_space(pos, q, pos, q, True)
        + _recip_space(pos, q, pos, q)
        + _k0_term(pos, q, pos, q)
        + _self_energy(q)
    )

    # --- real–mirror contribution (no self-exclusion, no self-energy) ---
    E_real_mirror = (
        _real_space(pos, q, pos_mirror, q_mirror, False)
        + _recip_space(pos, q, pos_mirror, q_mirror)
        + _k0_term(pos, q, pos_mirror, q_mirror)
    )

    return prefactor * (E_real_real + E_real_mirror)
