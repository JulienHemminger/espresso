import numpy as np
import espressomd
import espressomd.electrostatics

def _get_f_3d(system, params: dict):
    prefactor = params['prefactor']
    pw_err = 1e-8
    
    lx, ly, lz = system.box_l
    particles = system.part.all()

    # 1. 3D Periodic Forces from P3M
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    f_3d = np.array([p.f for p in particles])
    return f_3d

def _get_elc_correction(system, params):
    gap_size = params['gap_size']
    pw_err = 1e-8
    
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    qs = particles.q
    xs, ys, zs = particles.pos.T
    volume = lx * ly * lz

    # 2. Moments calculation
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)

    # 3. Non-Neutral / Dipole Force Correction
    f_corr_moments = np.zeros((n_part, 3))
    f_corr_moments[:, 2] = -(4.0 * np.pi / volume) * qs * (xi1 - xi0 * zs)

    # 4. Reciprocal Space ELC Correction
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_range = np.arange(-p_max, p_max + 1)
    q_range = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    pk, qk = P[mask], Q[mask]
    fx, fy = pk / lx, qk / ly
    f = np.sqrt(fx**2 + fy**2)

    arg_x = 2.0 * np.pi * fx
    arg_y = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f

    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    def get_chi(ez, tx, ty):
        return np.sum(qs[:, None] * ez * tx * ty, axis=0)

    chi_p = [get_chi(ex_p, cx, cy), get_chi(ex_p, sx, cy), get_chi(ex_p, cx, sy), get_chi(ex_p, sx, sy)]
    chi_m = [get_chi(ex_m, cx, cy), get_chi(ex_m, sx, cy), get_chi(ex_m, cx, sy), get_chi(ex_m, sx, sy)]

    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    term_pref = (1.0 / (lx * ly * f)) * rep

    f_elc_recip = np.zeros((n_part, 3))
    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        f_elc_recip[:, 0] += (qs[:, None] * (ex_p * dtx * ty * chi_m[i] + ex_m * dtx * ty * chi_p[i]) @ term_pref)
        f_elc_recip[:, 1] += (qs[:, None] * (ex_p * tx * dty * chi_m[i] + ex_m * tx * dty * chi_p[i]) @ term_pref)
        f_elc_recip[:, 2] += (qs[:, None] * arg_z * (ex_p * tx * ty * chi_m[i] - ex_m * tx * ty * chi_p[i]) @ term_pref)

    # Total Force Assembly
    return f_elc_recip + f_corr_moments


import numpy as np
import espressomd
import espressomd.electrostatics

# [_get_f_3d and _get_elc_correction remain identical to Step 1]

def _get_far_field_forces(system, params, idx_bulk, idx_bot, idx_top):
    """
    Computes the analytical far-field force contribution from image chains L_±2.
    Under Step 2B invariant validation (Delta_t = Delta_b = 0), this routine
    runs through the full math pipeline but structurally returns a clean zero matrix.
    """
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    
    qs = particles.q
    xs, ys, zs = particles.pos.T
    volume = lx * ly * lz
    
    # Extract reflection parameters
    dt = params.get('delta_mid_top', 0.0)
    db = params.get('delta_mid_bot', 0.0)
    delta = dt * db
    pw_err = 1e-8
    gap_size = params['gap_size']

    f_far = np.zeros((n_part, 3))

    # --- PART A: DIPOLE / MOMENT FAR-FIELD CONTRIBUTION ---
    # 1. Define the analytical I(z) polynomial sum function
    def get_I_sum(z_val):
        denom = 1.0 - delta
        if abs(denom) < 1e-15:
            return np.zeros_like(z_val)
        return (1.0 / denom) * (z_val + 2.0 * lz * delta / denom)

    # 2. Evaluate zeroth-order moments xi^(0) for L_-2 and L_+2
    xi0_real = np.sum(qs)
    xi1_real = np.sum(qs * zs)
    
    # Slices for L_-2 combinations
    xi0_m2 = (np.sum(qs[idx_bot]) * (db * delta + delta) + 
              np.sum(qs[idx_bulk]) * (db + delta) + 
              np.sum(qs[idx_top]) * (db + delta)) / (1.0 - delta)
              
    # Slices for L_+2 combinations
    xi0_p2 = (np.sum(qs[idx_top]) * (dt * delta + delta) + 
              np.sum(qs[idx_bulk]) * (dt + delta) + 
              np.sum(qs[idx_bot]) * (dt + delta)) / (1.0 - delta)

    # 3. Evaluate first-order moments xi^(1) for L_-2 and L_+2
    xi1_m2 = (np.sum(qs[idx_bot] * (-db * delta * get_I_sum(2.0 * lz + zs[idx_bot]) - delta * get_I_sum(2.0 * lz - zs[idx_bot]))) +
              np.sum(qs[idx_bulk] * (-db * get_I_sum(zs[idx_bulk]) - delta * get_I_sum(2.0 * lz - zs[idx_bulk]))) +
              np.sum(qs[idx_top] * (-db * get_I_sum(zs[idx_top]) - delta * get_I_sum(2.0 * lz - zs[idx_top]))))
              
    xi1_p2 = (np.sum(qs[idx_top] * (dt * delta * get_I_sum(4.0 * lz - zs[idx_top]) + delta * get_I_sum(2.0 * lz + zs[idx_top]))) +
              np.sum(qs[idx_bulk] * (dt * get_I_sum(2.0 * lz - zs[idx_bulk]) + delta * get_I_sum(2.0 * lz + zs[idx_bulk]))) +
              np.sum(qs[idx_bot] * (dt * get_I_sum(2.0 * lz - zs[idx_bot]) + delta * get_I_sum(2.0 * lz + zs[idx_bot]))))

    # Apply analytical background dipole force correction layer (Eq. 3.4 derivative)
    f_far[:, 2] += -(4.0 * np.pi / volume) * qs * (xi1_p2 - xi0_p2 * zs)  # Upper impact
    f_far[:, 2] += -(4.0 * np.pi / volume) * qs * (xi0_m2 * zs - xi1_m2)  # Lower impact

    # --- PART B: RECIPROCAL SPACE FAR-FIELD CONTRIBUTION ---
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_range = np.arange(-p_max, p_max + 1)
    q_range = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    pk, qk = P[mask], Q[mask]
    fx, fy = pk / lx, qk / ly
    f = np.sqrt(fx**2 + fy**2)

    arg_x = 2.0 * np.pi * fx
    arg_y = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f

    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    # Define analytical L_pq(z) series helper function
    def get_L_pq(z_val):
        return np.exp(-arg_z * z_val) / (1.0 - delta * np.exp(-2.0 * arg_z * lz))

    # Structural helper functions for gathering baseline real chi components
    def get_chi_base(ez_term, tx, ty):
        return np.sum(qs[:, None] * ez_term * tx * ty, axis=0)

    # 1. Generate structural image moments for upper layer L_+2
    # Combined terms for particles located in bulk, bottom, or top regions
    term_p2_z = np.zeros((n_part, len(f)))
    term_p2_z[idx_top] = dt * delta * get_L_pq(4.0 * lz - zs[idx_top, None]) + delta * get_L_pq(2.0 * lz + zs[idx_top, None])
    term_p2_z[idx_bulk] = dt * get_L_pq(2.0 * lz - zs[idx_bulk, None]) + delta * get_L_pq(2.0 * lz + zs[idx_bulk, None])
    term_p2_z[idx_bot] = dt * get_L_pq(2.0 * lz - zs[idx_bot, None]) + delta * get_L_pq(2.0 * lz + zs[idx_bot, None])

    # 2. Generate structural image moments for lower layer L_-2
    term_m2_z = np.zeros((n_part, len(f)))
    term_m2_z[idx_bot] = db * delta * get_L_pq(2.0 * lz + zs[idx_bot, None]) + delta * get_L_pq(2.0 * lz - zs[idx_bot, None])
    term_m2_z[idx_bulk] = db * get_L_pq(zs[idx_bulk, None]) + delta * get_L_pq(2.0 * lz - zs[idx_bulk, None])
    term_m2_z[idx_top] = db * get_L_pq(zs[idx_top, None]) + delta * get_L_pq(2.0 * lz - zs[idx_top, None])

    term_pref = (1.0 / (lx * ly * f))

    # Accumulate modes across 4 structural trigonometry indices
    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        # Calculate isolated collective chi parameters for images
        chi_p2 = np.sum(qs[:, None] * term_p2_z * tx * ty, axis=0)
        chi_m2 = np.sum(qs[:, None] * term_m2_z * tx * ty, axis=0)

        # Baseline real moments from ELC system
        chi_real_p = get_chi_base(np.exp(arg_z * zs[:, None]), cx, cy) 
        
        # Upper Image Field Force Execution (Image charges are above real charges)
        f_far[:, 0] += (qs[:, None] * (np.exp(arg_z * zs[:, None]) * dtx * ty) * chi_p2) @ term_pref
        f_far[:, 1] += (qs[:, None] * (np.exp(arg_z * zs[:, None]) * tx * dty) * chi_p2) @ term_pref
        f_far[:, 2] += (qs[:, None] * (arg_z * np.exp(arg_z * zs[:, None]) * tx * ty) * chi_p2) @ term_pref

        # Lower Image Field Force Execution (Image charges are below real charges)
        f_far[:, 0] += (qs[:, None] * (np.exp(-arg_z * zs[:, None]) * dtx * ty) * chi_m2) @ term_pref
        f_far[:, 1] += (qs[:, None] * (np.exp(-arg_z * zs[:, None]) * tx * dty) * chi_m2) @ term_pref
        f_far[:, 2] += (qs[:, None] * (-arg_z * np.exp(-arg_z * zs[:, None]) * tx * ty) * chi_m2) @ term_pref

    return f_far

def get_elcic_forces(system, params: dict):
    prefactor = params['prefactor']
    gap_size = params['gap_size']
    
    lambda_cutoff = params.get('lambda_cutoff', 2.0)
    if lambda_cutoff >= 0.5 * gap_size:
        lambda_cutoff = 0.1 * gap_size

    # --- PART 1: CLASSIFY REAL PARTICLES ---
    real_particles = system.part.all()
    n_real = len(real_particles)
    zs = np.array([p.pos[2] for p in real_particles])
    
    idx_bot = np.where(zs <= lambda_cutoff)[0]
    idx_top = np.where(zs >= (gap_size - lambda_cutoff))[0]
    idx_bulk = np.where((zs > lambda_cutoff) & (zs < (gap_size - lambda_cutoff)))[0]

    # --- PART 2: SOLVE COULOMB AND NEAR-FIELD BASELINE ---
    f_3d_total = _get_f_3d(system, params)
    f_elc_total = _get_elc_correction(system, params)
    f_near_real = (f_3d_total + prefactor * f_elc_total)[:n_real, :]

    # --- PART 3: ANALYTICAL FAR-FIELD MATRIX ENGINE ---
    f_far_real = _get_far_field_forces(system, params, idx_bulk, idx_bot, idx_top)

    return f_near_real + prefactor * f_far_real