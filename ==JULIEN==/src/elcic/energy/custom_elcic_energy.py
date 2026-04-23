import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy_contribs(
    gap_size, pw_error, system, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Computes ELCIC energy contributions for a slab system.
    Matches formulation in Tyagi et al., J. Chem. Phys. 129, 204102 (2008).
    """
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )

    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    
    # Delta (combined reflection coefficient) [cite: 771]
    delta = delta_mid_bot * delta_mid_top

    # 1. 3D Periodic Energy from P3M [cite: 833, 860]
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]

    # 2. Charge Moments (xi0, xi1, xi2) [cite: 854]
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)

    # 3. Handle Non-Neutral/Dipole Corrections [cite: 857, 858]
    # This matches Equation 3.10 for the dipole layer exchange
    fac = 2.0 * np.pi / (lx * ly * lz)
    e_non_neutral_corr = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    # 4. Reciprocal Space ELCIC Term [cite: 817, 857, 572]
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
    
    # Exponential factors for primary layer [cite: 797]
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    def s_term(ez, c1, c2):
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Chi factors (Fourier components) [cite: 797, 817]
    chi_p_cc, chi_m_cc = s_term(ex_p, cx, cy), s_term(ex_m, cx, cy)
    chi_p_sc, chi_m_sc = s_term(ex_p, sx, cy), s_term(ex_m, sx, cy)
    chi_p_cs, chi_m_cs = s_term(ex_p, cx, sy), s_term(ex_m, cx, sy)
    chi_p_ss, chi_m_ss = s_term(ex_p, sx, sy), s_term(ex_m, sx, sy)

    # Infinite sum of images L_pq(z) [cite: 572]
    # L_pq(z) = (exp(-2*pi*f*z)) / (1 - delta * exp(-4*pi*f*lz))
    denom = 1.0 - delta * np.exp(-2.0 * arg_z * lz)
    
    # Interaction factors including image reflections [cite: 572, 579, 585]
    # For ELCIC, we must sum the contributions of the image series
    # These terms modify the standard ELC factor (1 / (1 - exp(-arg_z * lz)))
    
    # Terms for top and bottom image series
    # Eq 4.6 and 4.10 describe the summation for multiple reflections
    term_bot = (delta_mid_bot * chi_m_cc + delta * chi_p_cc * np.exp(-arg_z * lz)) / denom
    term_top = (delta_mid_top * chi_p_cc * np.exp(-arg_z * lz) + delta * chi_m_cc) / denom
    
    # Energy correction (simplified for total sum of reciprocal components)
    # The ELCIC kernel relates the 2D+h system to the 3D system [cite: 857, 865]
    kernel = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    chi_base = (chi_p_cc * chi_m_cc + chi_p_sc * chi_m_sc + 
                chi_p_cs * chi_m_cs + chi_p_ss * chi_m_ss)
    
    e_recip = -np.sum((1.0 / (lx * ly * f)) * kernel * chi_base)

    return (float(prefactor), float(e_recip), float(e_3d), float(e_non_neutral_corr))

def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    _, e_recip, e_3d, e_non_neutral_corr = get_elcic_energy_contribs(
        gap_size, pw_error, system, prefactor, delta_mid_bot, delta_mid_top
    )
    # Total Energy = E_3D + E_dipole_corr + E_reciprocal_corr 
    return e_3d + (prefactor * e_non_neutral_corr) + (prefactor * e_recip)