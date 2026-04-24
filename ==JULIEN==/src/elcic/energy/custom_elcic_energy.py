import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    # --- 1. Setup P3M (The base 3D periodic solver) ---
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T

    # Current ESPResSo energy with 3D periodicity
    system.electrostatics.solver = p3m
    e_L0_3d = system.analysis.energy()["total"]

    # --- 2. Charge Moments (For Dipole/Non-neutral corrections) ---
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)
    
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    # Standard ELC constant term / dipole correction
    e_elc_const = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    # --- 3. Reciprocal Space Correction (ELC Sum) ---
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    def s_term(ez, c1, c2):
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    chi = (
        s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy)
        + s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy)
        + s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy)
        + s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy)
    )

    # Reciprocal term (standard ELC factor)
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    e_L0_elc_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)
    
    # Total ELC correction for the primary layer
    e_L0_elc = (prefactor * e_elc_const) + (prefactor * e_L0_elc_recip)
    e_L0_total = e_L0_3d + e_L0_elc

    # --- 4. Mapping to ELCIC structure ---
    # Since this is a template based on standard ELC, 
    # image layers (pm1, lt) and near-field (e_near) are placeholders 
    # or zeroed unless you implement the image charge summation logic.
    
    e_near = 0.0 # Typically short-range/real-space if separated
    e_far_total = e_L0_total # In plain ELC, far field is the total 3D+Corr
    
    # Placeholders for image charge layers (L=1, L=Total)
    e_L1_3d, e_L1_elc, e_L1_total = 0.0, 0.0, 0.0
    e_LT_3d, e_LT_elc, e_LT_total = 0.0, 0.0, 0.0
    e_far_detail = {} 

    contribs = {
        "e_near": float(e_near),
        "e_far":  float(e_far_total),
        "l0": {
            "e_3d":   float(e_L0_3d),
            "e_elc":  float(e_L0_elc),
            "total":  float(e_L0_total),
        },
        "pm1": {
            "e_3d":   float(e_L1_3d),
            "e_elc":  float(e_L1_elc),
            "total":  float(e_L1_total),
        },
        "lt": {
            "e_3d":   float(e_LT_3d),
            "e_elc":  float(e_LT_elc),
            "total":  float(e_LT_total),
        },
        "e_far_detail": e_far_detail,
    }
    return contribs


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    contribs = get_elcic_energy_contribs(
        system=system, 
        gap_size=gap_size, 
        pw_error=pw_error, 
        prefactor=prefactor, 
        delta_mid_bot=delta_mid_bot, 
        delta_mid_top=delta_mid_top
    )
    # Based on your requested return logic:
    return contribs["e_near"] + contribs["e_far"]