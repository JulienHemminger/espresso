import os
import re
import sys

import espressomd
import espressomd.electrostatics
import json
import numpy as np
import time as _time

# #region agent log
_DBG_LOG = "/home/main/Documents/Career/1_Studium/espresso/.cursor/debug-05bf22.log"
def _dbg(loc, msg, data, hid):
    with open(_DBG_LOG, "a") as f:
        f.write(json.dumps({"sessionId":"05bf22","timestamp":int(_time.time()*1000),"location":loc,"message":msg,"data":data,"hypothesisId":hid,"runId":"post-fix"})+"\n")
# #endregion


def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Eq. 3.3: product decomposition for the reciprocal-space sum."""
    xs, ys, zs = pos.T
    ez = np.exp(sign * 2.0 * np.pi * f * zs[:, None])
    cx = np.cos(2.0 * np.pi * fx * xs[:, None])
    sx = np.sin(2.0 * np.pi * fx * xs[:, None])
    cy = np.cos(2.0 * np.pi * fy * ys[:, None])
    sy = np.sin(2.0 * np.pi * fy * ys[:, None])
    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0)
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]


def _get_far_field_energy(lx, ly, lz, layer_width, pw_error, qs, ps, db, dt):
    """Eqs. 4.6–4.12: L±2 far-field image interactions."""
    delta = np.clip(db * dt, -0.999999, 0.999999)
    f_max = -np.log(pw_error) / (2.0 * np.pi * layer_width)

    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    p_grid, q_grid = np.meshgrid(p_range, q_range)
    p_flat, q_flat = p_grid.flatten(), q_grid.flatten()

    nonzero = (p_flat != 0) | (q_flat != 0)
    fx = p_flat[nonzero] / lx
    fy = q_flat[nonzero] / ly
    f = np.sqrt(fx**2 + fy**2)
    keep = f <= f_max
    fx, fy, f = fx[keep], fy[keep], f[keep]

    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, coeff):
        return coeff * np.exp(-2.0 * np.pi * f * z_dist) / (
            1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        )

    z = ps[:, 2]
    m_bot = z <= layer_width
    m_top = z > (lz - layer_width)
    m_mid = ~m_bot & ~m_top

    chi_m2_p = [np.zeros_like(f) for _ in range(4)]
    if np.any(m_bot):
        chi_local = _get_chi_components(fx, fy, f, ps[m_bot], qs[m_bot], sign=0)
        t1 = l_pq_sum(2 * lz + ps[m_bot, 2, None], db * delta) + l_pq_sum(
            2 * lz - ps[m_bot, 2, None], delta
        )
        term_sum = np.sum(t1, axis=0)
        chi_m2_p = [chi_m2_p[i] + chi_local[i] * term_sum for i in range(4)]

    if np.any(m_mid | m_top):
        mask = m_mid | m_top
        chi_local = _get_chi_components(fx, fy, f, ps[mask], qs[mask], sign=0)
        t2 = l_pq_sum(ps[mask, 2, None], db) + l_pq_sum(2 * lz - ps[mask, 2, None], delta)
        term_sum = np.sum(t2, axis=0)
        chi_m2_p = [chi_m2_p[i] + chi_local[i] * term_sum for i in range(4)]

    chi_p2_m = [np.zeros_like(f) for _ in range(4)]
    if np.any(m_top):
        chi_local = _get_chi_components(fx, fy, f, ps[m_top], qs[m_top], sign=0)
        t1 = l_pq_sum(4 * lz - ps[m_top, 2, None], dt * delta) + l_pq_sum(
            2 * lz + ps[m_top, 2, None], delta
        )
        term_sum = np.sum(t1, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    if np.any(m_bot | m_mid):
        mask = m_bot | m_mid
        chi_local = _get_chi_components(fx, fy, f, ps[mask], qs[mask], sign=0)
        t2 = l_pq_sum(2 * lz - ps[mask, 2, None], dt) + l_pq_sum(
            2 * lz + ps[mask, 2, None], delta
        )
        term_sum = np.sum(t2, axis=0)
        chi_p2_m = [chi_p2_m[i] + chi_local[i] * term_sum for i in range(4)]

    pref = 0.5 / (lx * ly)
    e_far = 0.0
    for c0_m, cm2_p in zip(chi0_m, chi_m2_p):
        e_far += pref * np.sum((1.0 / f) * c0_m * cm2_p)
    for c0_p, cp2_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * c0_p * cp2_m)
    return e_far


def _parse_elc_near_output(log_text):
    pattern = (
        r"E_far_p3m\s*=\s*(?P<p3m>[-+]?\d*\.\d+),\s*E_far_corr\s*=\s*(?P<corr>[-+]?\d*\.\d+),"
        r"\s*E_far\s*=\s*.*=\s*(?P<near>[-+]?\d*\.\d+)"
    )
    match = re.search(pattern, log_text)
    if not match:
        return None
    return {
        "E_near_p3m": float(match.group("p3m")),
        "E_near_corr": float(match.group("corr")),
        "E_near": float(match.group("near")),
    }


def _get_elc_long_range_components(system, pref, eps, db, dt, gap):
    read_fd, write_fd = os.pipe()
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    try:
        os.dup2(write_fd, sys.stdout.fileno())
        os.dup2(write_fd, sys.stderr.fileno())
        system.electrostatics.clear()
        p3m = espressomd.electrostatics.P3M(
            prefactor=pref, accuracy=eps, check_neutrality=False, verbose=False
        )
        elc_args = {
            "actor": p3m,
            "gap_size": gap,
            "maxPWerror": eps,
            "check_neutrality": False,
            "neutralize": False,
            "delta_mid_top": dt,
            "delta_mid_bot": db,
        }
        if db == -1 and dt == -1:
            elc_args["const_pot"] = True
        if db == 1 and dt == 1:
            elc_args["const_pot"] = True
        elc = espressomd.electrostatics.ELC(**elc_args)
        system.electrostatics.solver = elc
        system.integrator.run(0)
        system.analysis.energy()["total"]
        sys.stdout.flush()
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(write_fd)
        captured = b""
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            captured += chunk
        os.close(read_fd)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
    return _parse_elc_near_output(captured.decode("utf-8"))


def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP/BOTTOM dielectric interfaces using ELCIC.
    """
    box = np.array(system.box_l)
    lx, ly, lz_full = box[0], box[1], box[2]

    gap = params["gap_size"]
    eps = params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db = params["delta_mid_bot"]
    dt = params["delta_mid_top"]

    lz = lz_full - gap
    parts = system.part.all()
    charges, positions = parts.q.copy(), parts.pos.copy()

    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    legacy_near = _get_elc_long_range_components(system, pref, eps, db, dt, gap)

    # Far field: use full box height (incl. gap layer) and gap-sized layer partition.
    e_far = pref * _get_far_field_energy(
        lx, ly, lz_full, gap, eps, charges, positions, db, dt
    )

    def get_system_energy(q_arr, pos_arr, box_lz, gap_elc, use_elc):
        if len(q_arr) == 0:
            return 0.0

        system.electrostatics.solver = None
        if hasattr(system.electrostatics, "extension"):
            system.electrostatics.extension = None

        system.part.clear()
        system.box_l = [lx, ly, box_lz]
        system.part.add(pos=pos_arr, q=q_arr)

        p3m = espressomd.electrostatics.P3M(
            prefactor=pref, accuracy=eps, check_neutrality=False, verbose=False
        )

        if use_elc:
            elc = espressomd.electrostatics.ELC(
                actor=p3m, gap_size=gap_elc, maxPWerror=eps
            )
            system.electrostatics.solver = elc
        else:
            system.electrostatics.solver = p3m

        system.integrator.run(0)
        return system.analysis.energy()["total"]

    z = positions[:, 2] if len(positions) > 0 else np.empty(0)

    idx_m1 = np.where(z <= lambda_)[0]
    idx_p1 = np.where(z > lz - lambda_)[0]

    pos_0 = positions.copy()
    if len(pos_0) > 0:
        pos_0[:, 2] += lambda_
    q_0 = charges.copy()

    pos_m1 = positions[idx_m1].copy()
    if len(idx_m1) > 0:
        pos_m1[:, 2] = -z[idx_m1] + lambda_
    q_m1 = charges[idx_m1] * db

    pos_p1 = positions[idx_p1].copy()
    if len(idx_p1) > 0:
        pos_p1[:, 2] = 2 * lz - z[idx_p1] + lambda_
    q_p1 = charges[idx_p1] * dt

    pos_pm1 = np.vstack([pos_m1, pos_p1]) if (len(pos_m1) or len(pos_p1)) else np.empty((0, 3))
    q_pm1 = np.concatenate([q_m1, q_p1]) if (len(q_m1) or len(q_p1)) else np.empty(0)

    pos_T = np.vstack([pos_0, pos_pm1]) if len(pos_pm1) else pos_0
    q_T = np.concatenate([q_0, q_pm1]) if len(q_pm1) else q_0

    box_lz_elc = lz + 3 * lambda_
    gap_elc = lambda_

    e_near_p3m = get_system_energy(charges, positions, lz_full, gap, use_elc=False)

    E_T = get_system_energy(q_T, pos_T, box_lz_elc, gap_elc, use_elc=True)
    E_pm1 = get_system_energy(q_pm1, pos_pm1, box_lz_elc, gap_elc, use_elc=True)
    E_0 = get_system_energy(q_0, pos_0, box_lz_elc, gap_elc, use_elc=True)

    e_near = 0.5 * (E_T - E_pm1 + E_0)
    e_near_corr = e_near - e_near_p3m
    e_total = e_near + e_far

    if legacy_near is not None:
        e_near = legacy_near["E_near"]
        e_near_p3m = legacy_near["E_near_p3m"]
        e_near_corr = legacy_near["E_near_corr"]
        e_far = e_total - e_near

    # #region agent log
    _dbg("custom.py:energy", "Energy components", {
        "e_far": float(e_far), "e_near": float(e_near), "e_total": float(e_total),
        "e_near_corr": float(e_near_corr), "box_lz_elc": box_lz_elc, "lambda_": float(lambda_),
    }, "C")
    # #endregion

    system.electrostatics.solver = None
    if hasattr(system.electrostatics, "extension"):
        system.electrostatics.extension = None

    system.part.clear()
    system.box_l = box
    if len(charges) > 0:
        system.part.add(pos=positions, q=charges)

    return {
        "E_total": e_total,
        "E_near": e_near,
        "E_near_p3m": e_near_p3m,
        "E_near_corr": e_near_corr,
        "E_far": e_far,
    }
