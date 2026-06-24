import numpy as np
from scipy.special import erf, erfc

from elc.energy._2_small_box_neutral_dipole.reference_solution.ewald2d import (
    get_ewald_energy_2d
)


def direct_sum_energy(system, n_max=100, prefactor=1.0, eps=1.0, eps0=1.0):
    """
    Brute-force Coulomb energy for 2D periodic systems with non-neutral correction.

    Includes the Delta E term to handle systems where sum(q) != 0.

    U = q1 * q2 / r
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
