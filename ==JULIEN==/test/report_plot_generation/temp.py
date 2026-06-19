from elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
import espressomd
import math


NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor"]
OPTIONAL_KEYS = ["delta_mid_top", "delta_mid_bot"]


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def lerp_dict(start_params, end_params, t):
    """Modular interpolation of parameters."""
    lerp_params = {
        key: lerp(start_params[key], end_params[key], t) for key in NESSECARY_KEYS
    }

    for key in OPTIONAL_KEYS:
        if key in start_params and key in end_params:
            lerp_params[key] = lerp(start_params[key], end_params[key], t)

    lerp_params["positions"] = [
        lerp(np.array(p_start), np.array(p_end), t)
        for p_start, p_end in zip(start_params["positions"], end_params["positions"])
    ]
    lerp_params["charges"] = [
        lerp(q_start, q_end, t)
        for q_start, q_end in zip(start_params["charges"], end_params["charges"])
    ]
    return lerp_params



def get_E_recip(gap_size, pw_error, system, prefactor=1.0):
    lx, ly, lz = system.box_l
    particles = system.part.all()
    qs, (xs, ys, zs) = particles.q, particles.pos.T

    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)
    volume = lx * ly * lz

    # 4. Reciprocal Space ELC Term
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    # Exclude the k=0 mode (handled by the real space and dipole terms)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    # Particle-wise components
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Compute form factors (Chi) linearly
    def s_term(ez, c1, c2):
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Summing over the four combinations of sin/cos for the 2D Fourier transform
    chi = (
        s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy)
        + s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy)
        + s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy)
        + s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy)
    )

    # The reciprocal energy correction
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    E_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)

    return E_recip





start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0,
    "delta_mid_bot": 0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 9.999]),
        np.array([3, 2, 9.999]),
    ],  # legacy fails for part.z <= 3
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 15.0,  # legacy runs with: 14, 15, fails with 16, 20
    "prefactor": 1.0,
    "delta_mid_top": 0,
    "delta_mid_bot": 0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 30]),
        np.array([3, 2, 30]),
    ],  # legacy runs for part.z = 4, 24, fails for part.z = 3, 34, 39
}
end_params["lz"] = start_params["gap_size"] + 40

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

N = 5
energies = []
for t in np.linspace(0, 1, num=N):
    params = lerp_dict(start_params, end_params, t)

    system.electrostatics.clear()
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    energies.append(get_E_recip(params["gap_size"], 1e-8, system))

print(f"{N=}: {energies=}")

