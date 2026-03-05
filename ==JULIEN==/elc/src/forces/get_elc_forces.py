import numpy as np
import espressomd
import espressomd.electrostatics

def get_elc_forces(system, gap_size=1.0, pw_err=1e-6) -> list[np.ndarray]:
    lx, ly, lz = system.box_l
    h = lz - gap_size  # Real slab height
    ux, uy, uz = 1.0/lx, 1.0/ly, 1.0/lz
    
    particles = system.part.all()
    qs = particles.q
    pos = particles.pos
    n_part = len(qs)

    # 1. Compute 3D Periodic Forces using P3M
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    f_3d = np.copy(particles.f)

    # 2. ELC Correction: Dipole Term
    # Subtract P3M 3D dipole force and add ELC 2D+h dipole force
    total_dipole_z = np.sum(qs * pos[:, 2])
    # P3M 3D dipole force on particle i: (4*pi / 3*V) * q_i * M_z
    f_dip_3d_z = (4.0 * np.pi / (3.0 * lx * ly * lz)) * qs * total_dipole_z
    # ELC 2D+h dipole force: -4*pi * ux * uy * uz * q_i * M_z
    f_dip_elc_z = - (4.0 * np.pi * ux * uy * uz) * qs * total_dipole_z
    
    f_cor = np.zeros_like(f_3d)
    f_cor[:, 2] = - f_dip_3d_z + f_dip_elc_z

    # 3. ELC Correction: Fourier Summation
    # Determine the number of modes p, q based on accuracy and gap size
    # For simplicity in this implementation, we use a fixed range or 
    # a range derived from the gap exponential decay: exp(-2*pi*f_pq*gap)
    prec = int(np.ceil(-np.log(pw_err) / (2 * np.pi * gap_size * min(ux, uy))))
    R_limit = max(2, prec)

    for p in range(-R_limit, R_limit + 1):
        for q in range(-R_limit, R_limit + 1):
            if p == 0 and q == 0:
                continue
            
            fpq = np.sqrt((ux * p)**2 + (uy * q)**2)
            wp = 2 * np.pi * ux * p
            wq = 2 * np.pi * uy * q
            
            # Layer factor for replicated images in z
            # L_prime(z) = exp(-2*pi*fpq*z) / (1 - exp(-2*pi*fpq*lz))
            # Note: The paper uses 4*pi*Lz in some versions; for P3M gap it is 2*pi*fpq*Lz
            layer_fac = 1.0 / (1.0 - np.exp(-2 * np.pi * fpq * lz))
            
            # Components of structure factors (cc: cos*cos, sc: sin*cos, etc.)
            exp_pos = np.exp(2 * np.pi * fpq * pos[:, 2])
            exp_neg = np.exp(-2 * np.pi * fpq * pos[:, 2])
            cos_px = np.cos(wp * pos[:, 0])
            sin_px = np.sin(wp * pos[:, 0])
            cos_qy = np.cos(wq * pos[:, 1])
            sin_qy = np.sin(wq * pos[:, 1])

            # Pre-calculate common terms for efficiency
            q_cos_cos = qs * cos_px * cos_qy
            q_sin_cos = qs * sin_px * cos_qy
            q_cos_sin = qs * cos_px * sin_qy
            q_sin_sin = qs * sin_px * sin_qy

            # Structure factors chi (real charges) and X (replicas)
            # chi_plus uses exp(+2*pi*fpq*z), chi_minus uses exp(-2*pi*fpq*z)
            chi_p_cc = np.sum(q_cos_cos * exp_pos)
            chi_p_sc = np.sum(q_sin_cos * exp_pos)
            chi_p_cs = np.sum(q_cos_sin * exp_pos)
            chi_p_ss = np.sum(q_sin_sin * exp_pos)

            chi_m_cc = np.sum(q_cos_cos * exp_neg)
            chi_m_sc = np.sum(q_sin_cos * exp_neg)
            chi_m_cs = np.sum(q_cos_sin * exp_neg)
            chi_m_ss = np.sum(q_sin_sin * exp_neg)

            # Forces per particle i (Differentiating Eq 3.5)
            # F_z term involves -/+ 2*pi*fpq factor from gradient of exponential
            fac = (ux * uy / fpq) * layer_fac * np.exp(-2 * np.pi * fpq * lz)
            
            # Adding the force contribution to each particle
            # This is a manual gradient calculation of the chi*X terms
            term_z = (exp_pos * chi_m_cc + exp_neg * chi_p_cc) * cos_px * cos_qy + \
                     (exp_pos * chi_m_sc + exp_neg * chi_p_sc) * sin_px * cos_qy + \
                     (exp_pos * chi_m_cs + exp_neg * chi_p_cs) * cos_px * sin_qy + \
                     (exp_pos * chi_m_ss + exp_neg * chi_p_ss) * sin_px * sin_qy
            
            f_cor[:, 2] += qs * (2 * np.pi * fpq) * fac * term_z
            
            # X and Y forces involve derivatives of sin/cos
            f_cor[:, 0] += qs * fac * wp * (
                (exp_pos * chi_m_cc + exp_neg * chi_p_cc) * (-sin_px * cos_qy) +
                (exp_pos * chi_m_sc + exp_neg * chi_p_sc) * (cos_px * cos_qy)
            )
            f_cor[:, 1] += qs * fac * wq * (
                (exp_pos * chi_m_cc + exp_neg * chi_p_cc) * (cos_px * -sin_qy) +
                (exp_pos * chi_m_cs + exp_neg * chi_p_cs) * (cos_px * cos_qy)
            )

    return [f3 + fc for f3, fc in zip(f_3d, f_cor)]