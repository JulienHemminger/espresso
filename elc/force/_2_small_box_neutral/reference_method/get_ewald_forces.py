import numpy as np
from scipy.special import erf, erfc


def get_ewald_forces_2d(system, n_max=10, prefactor=1.0):
    """
    Computes forces for a 2D periodic system (slab geometry)
    using a direct Ewald summation approach.
    """
    positions = system.part.all().pos
    charges = system.part.all().q
    lx, ly = system.box_l[0], system.box_l[1]

    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)
    A = lx * ly
    eta = np.sqrt(np.pi) / min(lx, ly)

    forces = np.zeros_like(pos)

    # Precompute separation vectors and charge products
    # dr[i, j] = pos[i] - pos[j]
    dr = pos[:, None, :] - pos[None, :, :]
    qq = q[:, None] * q[None, :]

    # ---- 1. Real-space Forces ----
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            R = np.array([nx * lx, ny * ly, 0.0])
            rvec = dr + R  # (N, N, 3)
            dist = np.linalg.norm(rvec, axis=2)

            if nx == 0 and ny == 0:
                np.fill_diagonal(dist, np.inf)

            # Scalar part of the force: -d/dr [erfc(eta*r)/r]
            # = [erfc(eta*r)/r^2 + (2*eta/sqrt(pi))*exp(-eta^2*r^2)/r]
            exp_part = (2.0 * eta / np.sqrt(np.pi)) * np.exp(-((eta * dist) ** 2))
            scalar = erfc(eta * dist) / dist**2 + exp_part / dist

            # Vector force: scalar * (rvec / dist) * q_i * q_j
            f_pair = (qq * scalar / dist)[:, :, None] * rvec
            forces += np.sum(f_pair, axis=1)

    # ---- 2. Reciprocal-space Forces (G != 0) ----
    gx0, gy0 = 2.0 * np.pi / lx, 2.0 * np.pi / ly
    drho = dr[:, :, :2]
    dz = dr[:, :, 2]

    for mx in range(-n_max, n_max + 1):
        for my in range(-n_max, n_max + 1):
            if mx == 0 and my == 0:
                continue

            Gx, Gy = mx * gx0, my * gy0
            G = np.sqrt(Gx**2 + Gy**2)

            # Phase and h-function
            dot_prod = drho[:, :, 0] * Gx + drho[:, :, 1] * Gy
            arg_p = G / (2.0 * eta) + eta * dz
            arg_m = G / (2.0 * eta) - eta * dz

            exp_p, exp_m = np.exp(G * dz), np.exp(-G * dz)
            erfc_p, erfc_m = erfc(arg_p), erfc(arg_m)

            # In-plane forces (from d/drho cos(G.rho))
            # dE/drho_i = - (pi/A*G) * h * G * sin(G.rho)
            h = exp_p * erfc_p + exp_m * erfc_m
            f_scalar_xy = (np.pi / A) * qq * h * np.sin(dot_prod)
            forces[:, 0] += np.sum(f_scalar_xy * (Gx / G), axis=1)
            forces[:, 1] += np.sum(f_scalar_xy * (Gy / G), axis=1)

            # Out-of-plane forces (from d/dz h(G, dz))
            # dh/dz = G(exp(G*dz)erfc(arg_p) - exp(-G*dz)erfc(arg_m))
            dh_dz = G * (exp_p * erfc_p - exp_m * erfc_m)
            f_z = -(np.pi / (A * G)) * qq * dh_dz * np.cos(dot_prod)
            forces[:, 2] += np.sum(f_z, axis=1)

    # ---- 3. G = 0 Force Term ----
    # d/dz_i [ -pi/A * sum(qq * (z*erf(eta*z) + exp(-eta^2*z^2)/(eta*sqrt(pi)))) ]
    # The derivative is simple: -2*pi/A * q_i * sum(q_j * erf(eta * z_ij))
    f_g0_z = (2.0 * np.pi / A) * q[:, None] * (q[None, :] * erf(eta * dz))
    forces[:, 2] += np.sum(f_g0_z, axis=1)

    return forces * prefactor
