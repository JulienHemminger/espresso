import espressomd
import espressomd.electrostatics
import numpy as np


def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Product decomposition for the ELC reciprocal sum."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T

    # Use real/imaginary parts to represent sin/cos product decomposition
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])

    cx = np.cos(arg_x * xs[:, None])
    sx = np.sin(arg_x * xs[:, None])
    cy = np.cos(arg_y * ys[:, None])
    sy = np.sin(arg_y * ys[:, None])

    return [
        np.sum(qs[:, None] * ez * c1 * c2, axis=0)
        for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]
    ]


def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """
    Evaluates far image interactions optimized for a TOP interface (dt).
    Here, db is effectively 0, simplifying the far-field summation.
    """
    lx, ly, lz = box
    delta = db * dt
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    # Top far field logic (L+2) dominates when delta_bot is 0
    m_top = ps[:, 2] > 0  # All charges interact with the top interface
    chi_p2_m = [np.zeros_like(f) for _ in range(4)]

    if np.any(m_top):
        chi_local = _get_chi_components(fx, fy, f, ps[m_top], qs[m_top], sign=0)
        # Reflecting across top boundary: z_img = 2lz - z
        t = l_pq_sum(2 * lz - ps[m_top, 2, None], dt)
        term_sum = np.sum(t, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    pref = 0.5 / (lx * ly)
    e_far = 0.0
    # Cross terms between real charges and top-reflected images
    for c0_p, cp2_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * c0_p * cp2_m)

    return e_far


def _get_config_energy(system, p_set, q_set, prefactor, accuracy, gap_size, lz):
    lx, ly = system.box_l[0], system.box_l[1]
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_set, q=q_set)

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]

    # Correction term logic remains consistent with ELC non-neutrality
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_set[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    e_corr = prefactor * fac * (xi1**2 - (lz**2 / 12.0) * xi0**2)

    system.electrostatics.clear()
    return e_3d + e_corr


def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP dielectric interface.
    """
    box = np.array(system.box_l)
    lz = box[2]
    gap_size, eps = params["gap_size"], params["pw_error"]
    prefactor = params.get("prefactor", 1.0)
    delta_mid_bot, delta_mid_top = params["delta_mid_bot"], params["delta_mid_top"]

    particles = system.part.all()
    charges, positions = particles.q.copy(), particles.pos.copy()

    # Top Images
    positions_top_images = positions.copy()
    positions_top_images[:, 2] = 2 * (lz - gap_size) - positions_top_images[:, 2]
    charges_top_images = charges * delta_mid_top

    # 2. Near-Field Energy calculation
    E_l0 = _get_config_energy(system, positions, charges, prefactor, eps, gap_size, lz)
    E_pm1 = _get_config_energy(
        system, positions_top_images, charges_top_images, prefactor, eps, gap_size, lz
    )

    positions_total = np.vstack([positions, positions_top_images])
    charges_total = np.concatenate([charges, charges_top_images])
    E_lt = _get_config_energy(
        system, positions_total, charges_total, prefactor, eps, gap_size, lz
    )

    E_near = 0.5 * (E_lt - E_pm1 + E_l0)

    E_far = prefactor * _get_far_field_energy(
        box, gap_size, eps, charges, positions, delta_mid_bot, delta_mid_top
    )

    E_total = E_near + E_far
    return {
        "E_l0": E_l0,
        "E_pm1": E_pm1,
        "E_lt": E_lt,
        "E_near": E_near,
        "E_far": E_far,
        "E_total": E_total,
    }

import espressomd
import espressomd.electrostatics
from common.generators.position_generator import get_rdm_constrained_points_np
from elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d

import matplotlib.pyplot as plt
import numpy as np
from src.common.has_downward_trend import has_downward_trend
from common.generators.position_generator import get_rdm_constrained_points_np
import espressomd
import espressomd.electrostatics
import numpy as np
from src.elc.energy.legacy_elc_energy import get_legacy_energy
import copy
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 1]), np.array([3, 2, 15])],
    "pw_error": 1e-8,
    "delta_mid_bot": 0.0,
    "delta_mid_top": -1.0,
}
end_params = copy.deepcopy(start_params)
end_params["delta_mid_top"] = +1.0

system.part.clear()
system.box_l = [start_params["lx"], start_params["ly"], start_params["lz"]]
for i in range(len(start_params["charges"])):
    system.part.add(pos=start_params["positions"][i], q=start_params["charges"][i])


legacy_energy = get_legacy_energy(system=system, params_dict=start_params)
custom_energy_contribs = get_elcic_energy(system, start_params)

print(f"{legacy_energy=}")
print(f"{custom_energy_contribs=}")
"""
legacy_energy=-0.01670777087467143
custom_energy_contribs={'E_l0': np.float64(1.0065395656095748), 'E_pm1': np.float64(1.0065395656095746), 'E_lt': np.float64(0.6040646429898985), 'E_near': np.float64(0.30203232149494935), 'E_far': np.float64(1.065556484982957e+19), 'E_total': np.float64(1.065556484982957e+19)}

"""

STEPS = 4
# Write plotting code that lerps between start_params and end_params for "STEPS" and for each computes legacy and custom energy
# it plots legacy energy and custom_energy_contribs["E_total"] as solid lines.
# behind these lines is a bar chart. for every evaluation point, theres a bar chart behind where it shows the proportions of E_l0, E_pm1 and E_lt.