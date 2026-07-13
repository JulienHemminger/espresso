import numpy as np
from scipy.special import erf, erfc


def get_ewald_energy_2d(system, n_max=100, prefactor=1.0):
    pos = np.asarray(system.part.all().pos, dtype=np.float64)
    q = np.asarray(system.part.all().q, dtype=np.float64)

    lx, ly = system.box_l[0], system.box_l[1]
    area = lx * ly
    eta = np.sqrt(np.pi) / min(lx, ly)

    dr = pos[:, None, :] - pos[None, :, :]
    q_pairs = q[:, None] * q[None, :]

    # Real space component
    grid_range = np.arange(-n_max, n_max + 1)
    nx, ny = np.meshgrid(grid_range, grid_range, indexing="ij")
    rx_shifts = nx.flatten() * lx
    ry_shifts = ny.flatten() * ly

    e_real = 0.0
    for rx, ry in zip(rx_shifts, ry_shifts):
        r_vec = dr + np.array([rx, ry, 0.0])
        dist = np.linalg.norm(r_vec, axis=2)

        if rx == 0.0 and ry == 0.0:
            np.fill_diagonal(dist, np.inf)

        e_real += np.sum(q_pairs * erfc(eta * dist) / dist)
    e_real *= 0.5

    # Reciprocal space component
    kx_base = 2.0 * np.pi / lx
    ky_base = 2.0 * np.pi / ly
    mx, my = np.meshgrid(grid_range, grid_range, indexing="ij")
    mx, my = mx.flatten(), my.flatten()

    valid_g = (mx != 0) | (my != 0)
    gx = mx[valid_g] * kx_base
    gy = my[valid_g] * ky_base
    g = np.sqrt(gx**2 + gy**2)

    dr_xy = dr[:, :, :2]
    dz = dr[:, :, 2]
    abs_dz = np.abs(dz)

    e_recip = 0.0
    for k in range(len(g)):
        phase = dr_xy[:, :, 0] * gx[k] + dr_xy[:, :, 1] * gy[k]
        arg_plus = g[k] / (2.0 * eta) + eta * dz
        arg_minus = g[k] / (2.0 * eta) - eta * dz
        h_g = np.exp(g[k] * dz) * erfc(arg_plus) + np.exp(-g[k] * dz) * erfc(arg_minus)
        e_recip += np.sum(q_pairs * (np.pi / g[k]) * h_g * np.cos(phase))
    e_recip /= 2.0 * area

    # Self energy & G = 0 components
    e_self = -(eta / np.sqrt(np.pi)) * np.sum(q**2)

    g0_terms = np.where(
        abs_dz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        abs_dz * erf(eta * abs_dz)
        + np.exp(-((eta * abs_dz) ** 2)) / (eta * np.sqrt(np.pi)),
    )
    e_k0 = -(np.pi / area) * np.sum(q_pairs * g0_terms)

    return prefactor * (e_real + e_recip + e_k0 + e_self)
