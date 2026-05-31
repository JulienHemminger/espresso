import espressomd
import espressomd.electrostatics
import numpy as np

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    return 0

def _get_config_energy(system, p_set, q_set, prefactor, accuracy, lz):
    if len(q_set) == 0:
        return 0.0, 0.0 # Return tuple

    lx, ly = system.box_l[0], system.box_l[1]
    p_wrapped = p_set.copy()
    p_wrapped[:, 0] = np.mod(p_wrapped[:, 0], lx)
    p_wrapped[:, 1] = np.mod(p_wrapped[:, 1], ly)
    p_wrapped[:, 2] = np.mod(p_wrapped[:, 2], lz)
    
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_wrapped, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_wrapped[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    if np.isclose(xi0, 0.0, atol=1e-12):
        e_corr = prefactor * fac * (xi1**2)
    else:
        e_corr = 0.0
    
    system.electrostatics.clear()
    return e_3d, e_corr


def get_elcic_energy(system, params: dict):
    box = np.array(system.box_l)
    lz_full = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]
    lz = lz_full - gap
    
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    mask_bot = (ps_orig[:, 2] >= 0.0) & (ps_orig[:, 2] < lambda_)
    mask_top = (ps_orig[:, 2] > (lz - lambda_)) & (ps_orig[:, 2] <= lz)
    mask_mid = (ps_orig[:, 2] >= lambda_) & (ps_orig[:, 2] <= (lz - lambda_))

    ps_p1 = ps_orig[mask_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]
    qs_p1 = qs_orig[mask_top] * dt

    ps_m1 = ps_orig[mask_bot].copy()
    ps_m1[:, 2] = -ps_m1[:, 2]
    qs_m1 = qs_orig[mask_bot] * db

    ps_lt = np.vstack([ps_orig, ps_p1, ps_m1])
    qs_lt = np.concatenate([qs_orig, qs_p1, qs_m1])
    
    ps_pm1 = np.vstack([ps_p1, ps_m1])
    qs_pm1 = np.concatenate([qs_p1, qs_m1])
    
    e_l0_3d, e_l0_corr = _get_config_energy(system, ps_orig, qs_orig, pref, eps, lz_full)

    e_lt_3d, e_lt_corr = _get_config_energy(system, ps_lt, qs_lt, pref, eps, lz_full)

    e_pm1_3d, e_pm1_corr = _get_config_energy(system, ps_pm1, qs_pm1, pref, eps, lz_full)

    e_near_3d = 0.5 * (e_lt_3d - e_pm1_3d + e_l0_3d)
    e_near_corr = 0.5 * (e_lt_corr - e_pm1_corr + e_l0_corr)

    e_near = e_near_3d + e_near_corr
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)
    e_total = e_near + e_far

    return {
        "E_total": e_total,
        "E_near": e_near,
        "E_near_p3m": e_near_3d,
        "E_near_corr": e_near_corr,
        "E_far": e_far,
    }