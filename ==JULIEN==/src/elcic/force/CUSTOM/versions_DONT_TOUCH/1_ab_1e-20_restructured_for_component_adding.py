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

def get_elcic_forces(system, params: dict):
    prefactor = params['prefactor']
    
    # --- 1. CLASSIFICATION & VIRTUAL PARTICLE CREATION ---
    # (For Step 2, this will find 0 image particles, but structure it now)
    real_particles = system.part.all()
    n_real = len(real_particles)
    
    # TODO for Step 3/4/5: 
    # Determine lambda, check if any real particle z is near boundaries,
    # and system.part.add(...) virtual image charges with scaled charges.
    
    # Keep track of how many total particles exist now (real + virtual)
    # total_particles = system.part.all()
    
    # --- 2. PIPELINE EXECUTION ON THE EXPANDED SYSTEM ---
    # _get_f_3d executes P3M on whatever is currently inside `system`
    f_3d_total = _get_f_3d(system, params)
    
    # _get_elc_correction computes analytical ELC on whatever is inside `system`
    f_elc_total = _get_elc_correction(system, params)
    
    # Combine near-field forces
    f_near_total = f_3d_total + prefactor * f_elc_total
    
    # --- 3. FORCE FILTERING ---
    # Discard forces acting on virtual particles. We only care about 0:n_real
    f_near_real = f_near_total[:n_real, :]
    
    # --- 4. CLEANUP VIRTUAL PARTICLES ---
    # TODO for Step 3/4/5: Remove the added virtual particles from the ESPResSo system
    # so they don't corrupt the next integration step or duplicate in next evaluations.
    # e.g., for p in virtual_particles: p.remove()
    
    # --- 5. FAR-FIELD ANALYTICAL CORRECTION ---
    # For Step 2, you will implement the background infinite-image formula here.
    # It acts ONLY on the real particles using their coordinates.
    f_far_real = np.zeros((n_real, 3)) 
    
    if params["delta_mid_top"] != 0.0 or params["delta_mid_bot"] != 0.0:
        # TODO for Step 2: Implement the O(N) Far-Field vector sums here
        # f_far_real += compute_far_field_forces(real_particles, params)
        pass

    # Total physical force acting on the real system
    return f_near_real + f_far_real