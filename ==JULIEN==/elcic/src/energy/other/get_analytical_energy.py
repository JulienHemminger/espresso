import numpy as np


def calculate_elcic_energy(system, params, n_max=10):
    """
    Calculates ELCIC energy for a slab between two dielectric interfaces.
    n_max here refers to the number of image charge reflections (convergence is usually fast).
    """
    pos = np.asarray(system.part.all().pos, dtype=np.float64)
    q = np.asarray(system.part.all().q, dtype=np.float64)

    lx, ly = params["lx"], params["ly"]
    gap = params["gap_size"]
    d_top = params["delta_mid_top"]
    d_bot = params["delta_mid_bot"]
    pref = params["prefactor"]

    N = len(q)
    E_total = 0.0

    # 2D Periodic image vectors (reduced n_max for brute force efficiency)
    nx = np.arange(-n_max, n_max + 1)
    ny = np.arange(-n_max, n_max + 1)
    NX, NY = np.meshgrid(nx, ny)
    Rx, Ry = (NX.ravel() * lx), (NY.ravel() * ly)
    origin_idx = (2 * n_max + 1) * n_max + n_max

    for i in range(N):
        for j in range(N):
            dx = pos[i, 0] - pos[j, 0] + Rx
            dy = pos[i, 1] - pos[j, 1] + Ry

            # 1. Real-Real interaction (Standard 2D)
            dz0 = pos[i, 2] - pos[j, 2]
            dist0 = np.sqrt(dx**2 + dy**2 + dz0**2)
            if i == j:
                dist0[origin_idx] = np.inf
            E_total += q[i] * q[j] * np.sum(1.0 / dist0)

            # 2. Image Charge Summation
            # Only 1st order reflections shown for brevity;
            # for full ELCIC, you iterate through reflections k
            for k in range(1, 4):
                # Reflections depend on gap size and relative positions
                # Example: Bottom reflection
                dz_bot = pos[i, 2] + pos[j, 2] + 2 * (k - 1) * gap
                dist_bot = np.sqrt(dx**2 + dy**2 + dz_bot**2)
                E_total += q[i] * q[j] * (d_bot**k) * np.sum(1.0 / dist_bot)

                # Example: Top reflection
                dz_top = 2 * gap - (pos[i, 2] + pos[j, 2]) + 2 * (k - 1) * gap
                dist_top = np.sqrt(dx**2 + dy**2 + dz_top**2)
                E_total += q[i] * q[j] * (d_top**k) * np.sum(1.0 / dist_top)

    return 0.5 * pref * E_total
