import espressomd
import espressomd.electrostatics
import json
import numpy as np
import os
import re
import sys
import time as _time

from elc.energy.legacy_elc_energy import get_legacy_energy

# #region agent log
_DBG_LOG = "/home/main/Documents/Career/1_Studium/espresso/.cursor/debug-05bf22.log"


def _dbg(loc, msg, data, hid):
    def _ser(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    data = {k: _ser(v) for k, v in data.items()}
    with open(_DBG_LOG, "a") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "05bf22",
                    "timestamp": int(_time.time() * 1000),
                    "location": loc,
                    "message": msg,
                    "data": data,
                    "hypothesisId": hid,
                    "runId": "post-fix",
                }
            )
            + "\n"
        )


# #endregion


def _capture_legacy_near(system, params):
    read_fd, write_fd = os.pipe()
    orig_out, orig_err = os.dup(sys.stdout.fileno()), os.dup(sys.stderr.fileno())
    try:
        os.dup2(write_fd, sys.stdout.fileno())
        os.dup2(write_fd, sys.stderr.fileno())
        total = get_legacy_energy(system, params)
        sys.stdout.flush()
    finally:
        os.dup2(orig_out, sys.stdout.fileno())
        os.dup2(orig_err, sys.stderr.fileno())
        os.close(write_fd)
        captured = b""
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            captured += chunk
        os.close(read_fd)
        os.close(orig_out)
        os.close(orig_err)

    text = captured.decode("utf-8", errors="replace")
    pattern = (
        r"E_far_p3m\s*=\s*(?P<p3m>[-+]?\d*\.\d+),\s*"
        r"E_far_corr\s*=\s*(?P<corr>[-+]?\d*\.\d+),\s*"
        r"E_far\s*=\s*.*=\s*(?P<near>[-+]?\d*\.\d+)"
    )
    match = re.search(pattern, text)
    if not match:
        # #region agent log
        _dbg("custom.py:legacy_capture", "parse failed", {"tail": text[-500:]}, "I")
        # #endregion
        raise RuntimeError("Failed to parse legacy ELC near components")

    return {
        "E_total": total,
        "E_near_p3m": float(match.group("p3m")),
        "E_near_corr": float(match.group("corr")),
        "E_near": float(match.group("near")),
    }


def get_elcic_energy(system, params: dict):
    box = np.array(system.box_l)
    lx, ly, lz_full = box[0], box[1], box[2]
    gap = params["gap_size"]
    pw_error = params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db = params["delta_mid_bot"]
    dt = params["delta_mid_top"]
    delta = np.clip(db * dt, -0.99999, 0.99999)

    lz = lz_full - gap
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    parts = system.part.all()
    q_arr = parts.q.copy()
    pos_arr = parts.pos.copy()

    L0_neg1, L0_0, L0_pos1, L0 = [], [], [], []
    for q_i, p_i in zip(q_arr, pos_arr):
        L0.append((q_i, p_i.copy()))
        if p_i[2] <= lambda_:
            L0_neg1.append((q_i, p_i.copy()))
        elif p_i[2] <= lz - lambda_:
            L0_0.append((q_i, p_i.copy()))
        else:
            L0_pos1.append((q_i, p_i.copy()))

    L_neg1 = [(q_i * db, np.array([p[0], p[1], -p[2]])) for q_i, p in L0_neg1]
    L_pos1 = [
        (q_i * dt, np.array([p[0], p[1], 2 * lz - p[2]])) for q_i, p in L0_pos1
    ]


    shift_z = lambda_
    box_z_T = lz + 3 * lambda_
    gap_T = lambda_

    def shifted(lst):
        return [(qi, np.array([pi[0], pi[1], pi[2] + shift_z])) for qi, pi in lst]

    def calc_elc_energy(charge_pos, box_z_val, gap_val):
        if not charge_pos:
            return 0.0
        system.electrostatics.clear()
        system.part.clear()
        system.box_l = [lx, ly, box_z_val]
        for qi, pi in charge_pos:
            system.part.add(pos=pi, q=qi)
        p3m = espressomd.electrostatics.P3M(
            prefactor=pref,
            accuracy=pw_error,
            check_neutrality=False,
            verbose=False,
        )
        elc = espressomd.electrostatics.ELC(
            actor=p3m, gap_size=gap_val, maxPWerror=pw_error
        )
        system.electrostatics.solver = elc
        system.integrator.run(0)
        return system.analysis.energy()["total"]

    E_LT = calc_elc_energy(shifted(L_neg1 + L0 + L_pos1), box_z_T, gap_T)
    E_Lpm1 = calc_elc_energy(shifted(L_neg1 + L_pos1), box_z_T, gap_T)
    E_L0 = calc_elc_energy(shifted(L0), box_z_T, gap_T)
    e_near_elcic = 0.5 * (E_LT - E_Lpm1 + E_L0)

    def L_pq(z, f_pq):
        return np.exp(-2 * np.pi * f_pq * z) / (
            1.0 - delta * np.exp(-4 * np.pi * lz * f_pq)
        )

    def I_z(z):
        return (1.0 / (1.0 - delta)) * (z + 2 * lz * delta / (1.0 - delta))

    ux, uy = 1.0 / lx, 1.0 / ly
    K_cut = int(
        params.get(
            "K_cut",
            max(10, int(np.ceil(lx * uy * (-np.log(pw_error)) / (2 * np.pi * gap)))),
        )
    )

    xi_L0_0 = sum(qi for qi, _ in L0)
    xi_L0_1 = sum(qi * pi[2] for qi, pi in L0)

    xi_L_minus2_0, xi_L_minus2_1 = 0.0, 0.0
    for qi, pi in L0_neg1:
        xi_L_minus2_0 += (qi / (1 - delta)) * (db * delta + delta)
        xi_L_minus2_1 += (qi / (1 - delta)) * (
            -db * delta * I_z(2 * lz + pi[2]) - delta * I_z(2 * lz - pi[2])
        )
    for qi, pi in L0_0 + L0_pos1:
        xi_L_minus2_0 += (qi / (1 - delta)) * (db + delta)
        xi_L_minus2_1 += (qi / (1 - delta)) * (
            -db * I_z(pi[2]) - delta * I_z(2 * lz - pi[2])
        )

    xi_L_plus2_0, xi_L_plus2_1 = 0.0, 0.0
    for qi, pi in L0_pos1:
        xi_L_plus2_0 += (qi / (1 - delta)) * (dt * delta + delta)
        xi_L_plus2_1 += (qi / (1 - delta)) * (
            dt * delta * I_z(4 * lz - pi[2]) + delta * I_z(2 * lz + pi[2])
        )
    for qi, pi in L0_0 + L0_neg1:
        xi_L_plus2_0 += (qi / (1 - delta)) * (dt + delta)
        xi_L_plus2_1 += (qi / (1 - delta)) * (
            dt * I_z(2 * lz - pi[2]) + delta * I_z(2 * lz + pi[2])
        )

    sum_pq_plus2 = 0.0
    sum_pq_minus2 = 0.0

    for p in range(-K_cut, K_cut + 1):
        for q_idx in range(-K_cut, K_cut + 1):
            if p == 0 and q_idx == 0:
                continue
            f_pq = np.sqrt((p * ux) ** 2 + (q_idx * uy) ** 2)
            wp = 2 * np.pi * p * ux
            wq = 2 * np.pi * q_idx * uy

            cc0_p = sc0_p = cs0_p = ss0_p = 0.0
            cc0_m = sc0_m = cs0_m = ss0_m = 0.0
            for qi, pi in L0:
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                val_p = qi * np.exp(2 * np.pi * f_pq * pi[2])
                val_m = qi * np.exp(-2 * np.pi * f_pq * pi[2])
                cc0_p += val_p * cp * cq
                sc0_p += val_p * sp * cq
                cs0_p += val_p * cp * sq
                ss0_p += val_p * sp * sq
                cc0_m += val_m * cp * cq
                sc0_m += val_m * sp * cq
                cs0_m += val_m * cp * sq
                ss0_m += val_m * sp * sq

            cc_m2 = sc_m2 = cs_m2 = ss_m2 = 0.0
            for qi, pi in L0_neg1:
                val = qi * (
                    db * delta * L_pq(2 * lz + pi[2], f_pq)
                    + delta * L_pq(2 * lz - pi[2], f_pq)
                )
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_m2 += val * cp * cq
                sc_m2 += val * sp * cq
                cs_m2 += val * cp * sq
                ss_m2 += val * sp * sq
            for qi, pi in L0_0 + L0_pos1:
                val = qi * (
                    db * L_pq(pi[2], f_pq) + delta * L_pq(2 * lz - pi[2], f_pq)
                )
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_m2 += val * cp * cq
                sc_m2 += val * sp * cq
                cs_m2 += val * cp * sq
                ss_m2 += val * sp * sq

            cc_p2 = sc_p2 = cs_p2 = ss_p2 = 0.0
            for qi, pi in L0_pos1:
                val = qi * (
                    dt * delta * L_pq(4 * lz - pi[2], f_pq)
                    + delta * L_pq(2 * lz + pi[2], f_pq)
                )
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_p2 += val * cp * cq
                sc_p2 += val * sp * cq
                cs_p2 += val * cp * sq
                ss_p2 += val * sp * sq
            for qi, pi in L0_0 + L0_neg1:
                val = qi * (
                    dt * L_pq(2 * lz - pi[2], f_pq)
                    + delta * L_pq(2 * lz + pi[2], f_pq)
                )
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_p2 += val * cp * cq
                sc_p2 += val * sp * cq
                cs_p2 += val * cp * sq
                ss_p2 += val * sp * sq

            sum_pq_plus2 += (
                cc_p2 * cc0_p + sc_p2 * sc0_p + cs_p2 * cs0_p + ss_p2 * ss0_p
            ) / f_pq
            sum_pq_minus2 += (
                cc0_m * cc_m2 + sc0_m * sc_m2 + cs0_m * cs_m2 + ss0_m * ss_m2
            ) / f_pq

    Phi_plus2_half = 0.5 * ux * uy * sum_pq_plus2 - np.pi * ux * uy * (
        xi_L_plus2_1 * xi_L0_0 - xi_L_plus2_0 * xi_L0_1
    )
    Phi_minus2_half = 0.5 * ux * uy * sum_pq_minus2 - np.pi * ux * uy * (
        xi_L0_1 * xi_L_minus2_0 - xi_L0_0 * xi_L_minus2_1
    )
    e_far_elcic = pref * (Phi_plus2_half + Phi_minus2_half)
    e_total_elcic = e_near_elcic + e_far_elcic

    system.part.clear()
    system.box_l = [lx, ly, lz_full]
    for qi, pi in L0:
        system.part.add(pos=pi, q=qi)

    

  
    return {
        "E_total": e_total_elcic,
        "E_near": e_near_elcic,
        "E_near_p3m": 0,
        "E_near_corr": 0,
        "E_far": e_far_elcic,
    }
