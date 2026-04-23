import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy_contribs(
    gap_size, pw_error, system, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Implements ELCIC based on Tyagi et al., J. Chem. Phys. 129, 204102 (2008).
    """
    # 1. Setup P3M (Underlying 3D solver)
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]

    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    
    # Fundamental constants/parameters [cite: 61]
    ux, uy, uz = 1.0/lx, 1.0/ly, 1.0/lz
    delta = delta_mid_bot * delta_mid_top # [cite: 58]

    # 2. Charge Moments [cite: 67, 402]
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)

    # 3. Non-Neutral/Dipole Correction (Eq. 3.10) 
    # Note: 3D methods like P3M include a term proportional to (sum q_i r_i)^2.
    # ELC replaces this with a slab-specific correction.
    fac = 2.0 * np.pi * ux * uy * uz
    # Term: 2*pi*ux*uy*uz * [xi1^2 - xi0*xi2 - (lz^2/12)*xi0^2]
    e_dipole_corr = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * (xi0**2))

    # 4. Reciprocal Space ELCIC Term [cite: 365, 405]
    # Truncation radius based on gap_size and accuracy [cite: 62]
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    # Filter for non-zero vectors and f_max circle [cite: 62]
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P * ux)**2 + (Q * uy)**2) <= f_max)
    fx, fy = P[mask] * ux, Q[mask] * uy
    f = np.sqrt(fx**2 + fy**2)
    arg_z = 2.0 * np.pi * f
    
    # Exponential factors for product decomposition [cite: 67, 345]
    # chi_p = sum q*exp(2*pi*f*z)*exp(i*k.r)
    # chi_m = sum q*exp(-2*pi*f*z)*exp(i*k.r)
    exp_xy = np.exp(2j * np.pi * (fx * xs[:, None] + fy * ys[:, None]))
    chi_p = np.sum(qs[:, None] * np.exp(arg_z * zs[:, None]) * exp_xy, axis=0)
    chi_m = np.sum(qs[:, None] * np.exp(-arg_z * zs[:, None]) * exp_xy, axis=0)

    # Image-Corrected Kernels [cite: 76, 120]
    # Denominators for the geometric series of reflections
    # elc_denom is for replicas (1 - exp(-2*k*Lz)), elcic for images (1 - delta*exp(-2*k*Lz))
    denom_3d = 1.0 - np.exp(-2.0 * arg_z * lz)
    denom_ic = 1.0 - delta * np.exp(-2.0 * arg_z * lz)

    # Term 1: standard ELC part (Replicas) [cite: 365]
    # - (ux*uy) * sum (1/f) * [exp(-2*pi*f*lz)/denom_3d] * Re(chi_p * conj(chi_m))
    e_standard_elc = - (ux * uy) * np.sum(
        (1.0 / f) * (np.exp(-arg_z * lz) / denom_3d) * (chi_p * np.conj(chi_m)).real
    )
    
    # Term 2: Image charge series contribution [cite: 133, 135]
    # General expression for ELCIC reciprocal term (Simplified for L0-L0 case)
    term_image = (
        delta_mid_bot * (chi_m * np.conj(chi_m)).real + 
        delta_mid_top * (chi_p * np.conj(chi_p)).real * np.exp(-2.0 * arg_z * lz) +
        2.0 * delta * (chi_p * np.conj(chi_m)).real * np.exp(-arg_z * lz)
    ) / denom_ic
    
    e_image_corr = - (0.5 * ux * uy) * np.sum((1.0 / f) * term_image)

    e_recip = e_standard_elc + e_image_corr

    return (float(prefactor), float(e_recip), float(e_3d), float(e_dipole_corr))

def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    _, e_recip, e_3d, e_dipole_corr = get_elcic_energy_contribs(
        gap_size, pw_error, system, prefactor, delta_mid_bot, delta_mid_top
    )
    # Total Energy [cite: 75, 405]
    return e_3d + prefactor * (e_dipole_corr + e_recip)