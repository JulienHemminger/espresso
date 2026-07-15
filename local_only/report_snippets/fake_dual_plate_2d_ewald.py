import numpy as np
from scipy.special import erf, erfc, erfcx


def generate_image_charges(pos, q, H, eps_m, eps_b, eps_t, M):
    gamma_b = (eps_m - eps_b) / (eps_m + eps_b)
    gamma_t = (eps_m - eps_t) / (eps_m + eps_t)

    all_pos = [pos.copy()]
    all_q = [q.copy()]
    all_real = [np.ones(len(q), dtype=bool)]

    active_layers = [(pos.copy(), q.copy(), "none")]

    for _ in range(M):
        next_layers = []
        for p, charges, last_ref in active_layers:
            if last_ref != "b" and abs(gamma_b) > 1e-12:
                p_b = p.copy()
                p_b[:, 2] = -p[:, 2]
                q_b = charges * gamma_b
                all_pos.append(p_b)
                all_q.append(q_b)
                all_real.append(np.zeros(len(q), dtype=bool))
                next_layers.append((p_b, q_b, "b"))

            if last_ref != "t" and abs(gamma_t) > 1e-12:
                p_t = p.copy()
                p_t[:, 2] = 2.0 * H - p[:, 2]
                q_t = charges * gamma_t
                all_pos.append(p_t)
                all_q.append(q_t)
                all_real.append(np.zeros(len(q), dtype=bool))
                next_layers.append((p_t, q_t, "t"))

        if not next_layers:
            break
        active_layers = next_layers

    pos_all = np.concatenate(all_pos, axis=0)
    q_all = np.concatenate(all_q, axis=0)
    is_real = np.concatenate(all_real, dtype=bool)

    return pos_all, q_all, is_real


def get_ewald_energy_2d_icm(
    system,
    H,
    eps_m=1.0,
    eps_b=1.0,
    eps_t=1.0,
    M=4,
    n_max=50,
    prefactor=1.0,
):
    pos_real = np.asarray(system.part.all().pos, dtype=np.float64)
    q_real = np.asarray(system.part.all().q, dtype=np.float64)
    n_real = len(q_real)

    pos_all, q_all, is_real = generate_image_charges(
        pos_real, q_real, H, eps_m, eps_b, eps_t, M
    )

    lx, ly = system.box_l[0], system.box_l[1]
    area = lx * ly
    eta = np.sqrt(np.pi) / min(lx, ly)

    dr = pos_real[:, None, :] - pos_all[None, :, :]
    q_pairs = q_real[:, None] * q_all[None, :]

    grid_range = np.arange(-n_max, n_max + 1)
    nx, ny = np.meshgrid(grid_range, grid_range, indexing="ij")
    rx_shifts = nx.flatten() * lx
    ry_shifts = ny.flatten() * ly

    e_real = 0.0
    for rx, ry in zip(rx_shifts, ry_shifts):
        r_vec = dr + np.array([rx, ry, 0.0])
        dist = np.linalg.norm(r_vec, axis=2)

        if rx == 0.0 and ry == 0.0:
            np.fill_diagonal(dist[:, :n_real], np.inf)

        e_real += np.sum(q_pairs * erfc(eta * dist) / dist)

    e_real *= 0.5

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

    e_recip = 0.0
    g_2eta = g / (2.0 * eta)

    for k in range(len(g)):
        phase = dr_xy[:, :, 0] * gx[k] + dr_xy[:, :, 1] * gy[k]

        arg_plus = g_2eta[k] + eta * dz
        arg_minus = g_2eta[k] - eta * dz

        gaussian_decay = np.exp(-((g_2eta[k]) ** 2) - (eta * dz) ** 2)
        h_g = gaussian_decay * (erfcx(arg_plus) + erfcx(arg_minus))

        e_recip += np.sum(q_pairs * (np.pi / g[k]) * h_g * np.cos(phase))

    e_recip /= 2.0 * area

    e_self = -(eta / np.sqrt(np.pi)) * np.sum(q_real**2)

    abs_dz = np.abs(dz)
    g0_terms = np.where(
        abs_dz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        abs_dz * erf(eta * abs_dz)
        + np.exp(-((eta * abs_dz) ** 2)) / (eta * np.sqrt(np.pi)),
    )
    e_k0 = -(np.pi / area) * np.sum(q_pairs * g0_terms)

    return prefactor * (e_real + e_recip + 0.5 * e_k0 + e_self)
