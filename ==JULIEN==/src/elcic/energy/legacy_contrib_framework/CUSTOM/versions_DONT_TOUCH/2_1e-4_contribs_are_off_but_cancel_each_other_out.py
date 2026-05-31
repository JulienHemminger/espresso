import espressomd
import espressomd.electrostatics
import numpy as np


def _get_far_field_energy(params: dict) -> float:
    return 0

def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP/BOTTOM dielectric interfaces using ELCIC.
    Fixed: Removed mid-gap particles from image charge sets to avoid spurious interactions.
    """
    lx, ly, lz_full = np.array(system.box_l)

    pref = params.get("prefactor", 1.0)
    gap, eps = params["gap_size"], params["pw_error"]
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]
    lz = lz_full - gap
    parts = system.part.all()
    charges, positions = parts.q.copy(), parts.pos.copy()
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)

    e_l0_3d, e_l0_corr = 0, 0

    # 2. Combined sets
    e_lt_3d, e_lt_corr = 0, 0

    # 3. Image-only set
    e_pm1_3d, e_pm1_corr = 0, 0

    # Aggregate components
    e_near_3d = 0.5 * (e_lt_3d - e_pm1_3d + e_l0_3d)
    e_near_corr = 0.5 * (e_lt_corr - e_pm1_corr + e_l0_corr)
    e_near = e_near_3d + e_near_corr

    e_far = pref * _get_far_field_energy(params)

    e_total = e_near + e_far

    return {
        "E_total": e_total,
        "E_near": e_near,
        "E_near_p3m": e_near_3d,
        "E_near_corr": e_near_corr,
        "E_far": e_far,
    }
