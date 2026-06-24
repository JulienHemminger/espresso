import numpy as np
from scipy.special import erfcx, erf, erfc
import numpy as np
from scipy.special import erf, erfc



def get_ewald_energy_2d(params, n_max=100, k_max=10, tol=1e-8):
    positions = params["positions"]
    charges = params["charges"]
    prefactor = params["prefactor"]

    # 2. Get box dimensions (assuming a rectangular box)
    lx = params["lx"]
    ly = params["ly"]

    eta = None
    n_real = n_max
    n_recip = n_max
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    A = lx * ly

    if eta is None:
        # Balance real/reciprocal convergence
        eta = np.sqrt(np.pi) / min(lx, ly)

    # Pair separation vectors: dr[a, b] = pos[a] - pos[b]
    dr = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
    qq = q[:, None] * q[None, :]  # (N, N)

    # ---- Real-space sum ----
    E_real = 0.0
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            R = np.array([nx * lx, ny * ly, 0.0])
            rvec = dr + R  # (N, N, 3)
            dist = np.linalg.norm(rvec, axis=2)  # (N, N)

            if nx == 0 and ny == 0:
                np.fill_diagonal(dist, np.inf)  # exclude self

            contrib = qq * erfc(eta * dist) / dist
            E_real += np.sum(contrib)
    E_real *= 0.5

    # ---- Reciprocal-space sum (G != 0) ----
    gx0 = 2.0 * np.pi / lx
    gy0 = 2.0 * np.pi / ly
    drho = dr[:, :, :2]  # in-plane (N, N, 2)
    dz = dr[:, :, 2]  # z-separation (N, N)

    E_recip = 0.0
    for mx in range(-n_recip, n_recip + 1):
        for my in range(-n_recip, n_recip + 1):
            if mx == 0 and my == 0:
                continue
            Gx = mx * gx0
            Gy = my * gy0
            G = np.sqrt(Gx**2 + Gy**2)

            phase = drho[:, :, 0] * Gx + drho[:, :, 1] * Gy  # (N, N)

            # h(G, dz) = exp(G*dz)*erfc(G/(2*eta) + eta*dz)
            #           + exp(-G*dz)*erfc(G/(2*eta) - eta*dz)
            arg_p = G / (2.0 * eta) + eta * dz
            arg_m = G / (2.0 * eta) - eta * dz
            h = np.exp(G * dz) * erfc(arg_p) + np.exp(-G * dz) * erfc(arg_m)

            E_recip += np.sum(qq * (np.pi / G) * h * np.cos(phase))

    E_recip /= 2.0 * A

    # ---- Self-energy correction ----
    E_self = -(eta / np.sqrt(np.pi)) * np.sum(q**2)

    # ---- G = 0 term ----
    # Limit for |dz| -> 0:  |dz|*erf(eta*|dz|) + exp(-(eta*dz)^2)/(eta*sqrt(pi))
    #                      -> 1/(eta*sqrt(pi))
    adz = np.abs(dz)
    g0_terms = np.where(
        adz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        adz * erf(eta * adz) + np.exp(-((eta * adz) ** 2)) / (eta * np.sqrt(np.pi)),
    )
    E_G0 = -np.pi / A * np.sum(qq * g0_terms)

    E_total = E_real + E_recip + E_self + E_G0
    return E_total * prefactor


def direct_sum_energy(system, n_max=100, prefactor=1.0, eps=1.0, eps0=1.0):
    """
    Brute-force Coulomb energy for 2D periodic systems with non-neutral correction.

    Includes the Delta E term to handle systems where sum(q) != 0.
    """
    positions = system.part.all().pos
    charges = system.part.all().q
    lx = system.box_l[0]
    ly = system.box_l[1]
    lz = system.box_l[2]  # Often used as 'h' in ELC context

    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)
    Q_tot = np.sum(q)

    # 1. Standard Brute Force Sum
    nx = np.arange(-n_max, n_max + 1)
    ny = np.arange(-n_max, n_max + 1)
    NX, NY = np.meshgrid(nx, ny, indexing="ij")
    Rx = (NX.ravel() * lx).astype(np.float64)
    Ry = (NY.ravel() * ly).astype(np.float64)

    idx_origin = n_max * (2 * n_max + 1) + n_max

    E_sum = 0.0
    for a in range(N):
        # Vectorized over all images M for every pair (a, b)
        for b in range(N):
            dx_ab = pos[a, 0] - pos[b, 0] + Rx
            dy_ab = pos[a, 1] - pos[b, 1] + Ry
            dz_ab = pos[a, 2] - pos[b, 2]

            dist = np.sqrt(dx_ab**2 + dy_ab**2 + dz_ab**2)

            if a == b:
                dist[idx_origin] = np.inf  # Exclude self-interaction in the (0,0) cell

            E_sum += q[a] * q[b] * np.sum(1.0 / dist)

            E_sum -= (Q_tot**2 / (2 * lx * ly * eps0 * eps)) * lz

    # Apply 0.5 factor for pair counting
    # Note: In MD units, 1/(4*pi*eps0) is usually the 'prefactor'

    E_direct = 0.5 * E_sum

    return prefactor * E_direct
