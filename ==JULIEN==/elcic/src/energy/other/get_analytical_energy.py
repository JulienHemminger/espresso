import numpy as np


def calculate_elcic_energy(system, params, image_charge_reflection_count=10):
    """
    Calculates ELCIC energy for a slab between two dielectric interfaces.
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
    nx = np.arange(-image_charge_reflection_count, image_charge_reflection_count + 1)
    ny = np.arange(-image_charge_reflection_count, image_charge_reflection_count + 1)
    NX, NY = np.meshgrid(nx, ny)
    Rx, Ry = (NX.ravel() * lx), (NY.ravel() * ly)
    origin_idx = (
        2 * image_charge_reflection_count + 1
    ) * image_charge_reflection_count + image_charge_reflection_count

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

            # 2. Image Charge Summation (only 1st order reflections for now)
            for k in range(1, 4):
                dz_bot = pos[i, 2] + pos[j, 2] + 2 * (k - 1) * gap
                dist_bot = np.sqrt(dx**2 + dy**2 + dz_bot**2)
                E_total += q[i] * q[j] * (d_bot**k) * np.sum(1.0 / dist_bot)

                dz_top = 2 * gap - (pos[i, 2] + pos[j, 2]) + 2 * (k - 1) * gap
                dist_top = np.sqrt(dx**2 + dy**2 + dz_top**2)
                E_total += q[i] * q[j] * (d_top**k) * np.sum(1.0 / dist_top)

    return 0.5 * pref * E_total
