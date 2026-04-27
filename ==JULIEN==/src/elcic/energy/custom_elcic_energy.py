import espressomd
import espressomd.electrostatics
import numpy as np


def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    # --- 1. Setup P3M and Primary Layer (L0) ---
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T

    system.electrostatics.solver = p3m
    e_L0_3d = system.analysis.energy()["total"]

    # Charge Moments for L0
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    
    # ELC correction for L0 [cite: 91]
    e_elc_const_L0 = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    
    # --- 2. Reciprocal Space Setup ---
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    p_max, q_max = int(np.ceil(f_max * lx)), int(np.ceil(f_max * ly))
    p, q = np.arange(-p_max, p_max + 1), np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx)**2 + (Q / ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f

    def get_chi_terms(q_vec, z_vec):
        cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
        cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
        ex_p, ex_m = np.exp(arg_z * z_vec[:, None]), np.exp(-arg_z * z_vec[:, None])
        
        def s(ez, c1, c2): return np.sum(q_vec[:, None] * ez * c1 * c2, axis=0)
        
        # Returns Chi components (plus and minus) 
        return {
            "p": [s(ex_p, cx, cy), s(ex_p, sx, cy), s(ex_p, cx, sy), s(ex_p, sx, sy)],
            "m": [s(ex_m, cx, cy), s(ex_m, sx, cy), s(ex_m, cx, sy), s(ex_m, sx, sy)]
        }

    chi_L0 = get_chi_terms(qs, zs)
    
    # Standard ELC factor for primary layer [cite: 76]
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    chi_val_L0 = sum(p_i * m_i for p_i, m_i in zip(chi_L0["p"], chi_L0["m"]))
    e_L0_elc_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi_val_L0)
    
    e_L0_elc = prefactor * (e_elc_const_L0 + e_L0_elc_recip)
    e_L0_total = e_L0_3d + e_L0_elc

    """=========================================================="""
    # --- 3. Image Charge Interaction (L0 <-> L_pm1) ---
    # With single bottom interface, images are at -zs with charge delta*qs 
    qs_img = delta_mid_bot * qs
    zs_img = -zs 
    
    # Dipole/Constant Correction for Image Interaction 
    # FIXED: The ELC interaction term between L0 and an image layer L_img
    # requires a specific sign and prefactor to account for the slab geometry.
    xi0_img, xi1_img = np.sum(qs_img), np.sum(qs_img * zs_img)
    
    # This term accounts for the uniform background and the L0-Image dipole
    e_img_const = fac * (xi1 * xi1_img) # Note: fac was defined as 2*pi/volume

    # Reciprocal Interaction using Far Formula
    chi_img = get_chi_terms(qs_img, zs_img)
    
    # For images below the primary layer (z_img < z_L0), the interaction 
    # uses the Chi_minus of the top layer and Chi_plus of the bottom layer.
    chi_inter = sum(m_L0 * p_img for m_L0, p_img in zip(chi_L0["m"], chi_img["p"]))
    
    # FIXED: The normalization factor for the reciprocal interaction between 
    # two distinct layers in 2D Ewald is 2 * pi / (Lx * Ly)
    e_img_recip = np.sum((2.0 * np.pi / (lx * ly * arg_z)) * chi_inter)
    
    e_pm1_total = prefactor * (e_img_const + e_img_recip)
    """=========================================================="""

    return {
        "e_near": 0.0,
        "e_far": float(e_L0_total + e_pm1_total),
        "l0": {"e_3d": float(e_L0_3d), "e_elc": float(e_L0_elc), "total": float(e_L0_total)},
        "pm1": {"e_3d": 0.0, "e_elc": 0.0, "total": float(e_pm1_total)},
        "lt": {"e_3d": 0.0, "e_elc": 0.0, "total": 0.0},
        "e_far_detail": {}
    }

def get_elcic_energy(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top):
    contribs = get_elcic_energy_contribs(system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top)
    return contribs["e_near"] + contribs["e_far"]