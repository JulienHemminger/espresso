import numpy as np

def get_elc_energy(p3m, gap_size, pw_error, system):
    # https://gemini.google.com/app/11ddd9678a32b95b
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q
    pos = parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # 0. Activate P3M and obtain the 3D periodic energy (spherical summation)
    system.electrostatics.solver = p3m
    # We fetch the total energy; the correction will be applied to the Coulomb component
    e_3d = system.analysis.energy()["total"]
    
    # Prefactor C (Bjerrum length l_B * kT)
    prefactor = p3m.prefactor
    
    # Geometric parameters
    vol = lx * ly * lz
    ux, uy, uz = 1.0/lx, 1.0/ly, 1.0/lz
    
    # 1. Dipole Term Correction
    # Calculate dipole moments for the primary cell
    mx, my, mz = np.sum(qs * xs), np.sum(qs * ys), np.sum(qs * zs)
    
    # Correction: Replace spherical dipole term with slab dipole term
    # E_corr_dipole = C * [2*pi/V * Mz^2 - 2*pi/(3V) * (Mx^2 + My^2 + Mz^2)]
    e_corr_dipole = (2.0 * np.pi / vol) * (mz**2 - (mx**2 + my**2 + mz**2) / 3.0)
    e_corr_dipole *= prefactor
    
    # 2. Reciprocal Correction (Subtraction of Ghost Replicas)
    e_corr_recip = 0.0
    
    # Determine the reciprocal space cutoff (p_max, q_max) based on pw_error and gap
    # The ELC error decays as exp(-2*pi * f * gap_size)
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    # Iterate over reciprocal space modes (p, q)
    for p in range(-p_max, p_max + 1):
        for q in range(-q_max, q_max + 1):
            if p == 0 and q == 0:
                continue
            
            # Reciprocal frequency magnitude
            f = np.sqrt((p * ux)**2 + (q * uy)**2)
            if f > f_max:
                continue
            
            # Frequency components and trigonometric arguments
            omega_p = 2.0 * np.pi * p * ux
            omega_q = 2.0 * np.pi * q * uy
            arg_x = omega_p * xs
            arg_y = omega_q * ys
            arg_z = 2.0 * np.pi * f * zs
            
            # Compute structural sum factors (chi factors) for 4 trig combinations
            # Factors are sums of: q_i * exp(+- 2pi f zi) * {sin/cos}(wx xi) * {sin/cos}(wy yi)
            cos_x, sin_x = np.cos(arg_x), np.sin(arg_x)
            cos_y, sin_y = np.cos(arg_y), np.sin(arg_y)
            expp, expm = np.exp(arg_z), np.exp(-arg_z)
            
            def get_chi_product(qs, ez_p, ez_m, cx, sx, cy, sy):
                s_cc_p, s_cc_m = np.sum(qs*ez_p*cx*cy), np.sum(qs*ez_m*cx*cy)
                s_sc_p, s_sc_m = np.sum(qs*ez_p*sx*cy), np.sum(qs*ez_m*sx*cy)
                s_cs_p, s_cs_m = np.sum(qs*ez_p*cx*sy), np.sum(qs*ez_m*cx*sy)
                s_ss_p, s_ss_m = np.sum(qs*ez_p*sx*sy), np.sum(qs*ez_m*sx*sy)
                return s_cc_p*s_cc_m + s_sc_p*s_sc_m + s_cs_p*s_cs_m + s_ss_p*s_ss_m

            chi_prod = get_chi_product(qs, expp, expm, cos_x, sin_x, cos_y, sin_y)
            
            # Analytic geometric series sum for m!= 0 ghost replicas
            # a / (1 - a) where a = exp(-2*pi * f * lz)
            replica_factor = np.exp(-2.0 * np.pi * f * lz) / (1.0 - np.exp(-2.0 * np.pi * f * lz))
            
            # Energy correction for this mode: - (ux * uy / f) * replica_factor * chi_prod
            e_corr_recip -= (ux * uy / f) * replica_factor * chi_prod
            
    return e_3d + e_corr_dipole + prefactor * e_corr_recip