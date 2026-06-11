import numpy as np
import espressomd
import espressomd.electrostatics

def _get_f_3d(system, params: dict, active_particles_indices=None):
    """
    Computes standard 3D periodic forces using ESPResSo's P3M.
    NOTE FOR CURSOR: In ELCIC Near-field, 'system' will contain the real particles 
    PLUS the temporary primary image charges layer. Ensure you read forces from 
    the right particle instances.
    """
    prefactor = params['prefactor']
    pw_err = params.get('pw_error', 1e-8)
    
    # Run P3M integration
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    # If active_particles_indices is specified, extract only those forces
    particles = system.part.all()
    if active_particles_indices is not None:
        f_3d = np.array([particles[idx].f for idx in active_particles_indices])
    else:
        f_3d = np.array([p.f for p in particles])
    
    # Clean up solver to allow reuse/re-instantiation safely
    system.electrostatics.clear()
    return f_3d

def _get_elc_correction(system, params, active_particles_indices=None):
    """
    Computes the regular reciprocal and moment space corrections.
    NOTE FOR CURSOR: For ELCIC Near-field, the coordinates and charges fed here
    must include the real particles AND the explicit primary images (L_T).
    """
    gap_size = params['gap_size']
    pw_err = params.get('pw_error', 1e-8)
    
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    qs = particles.q
    xs, ys, zs = particles.pos.T
    volume = lx * ly * lz

    # Moments calculation
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)

    # Non-Neutral / Dipole Force Correction
    f_corr_moments = np.zeros((n_part, 3))
    f_corr_moments[:, 2] = -(4.0 * np.pi / volume) * qs * (xi1 - xi0 * zs)

    # Reciprocal Space ELC Correction
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

    total_elc = f_elc_recip + f_corr_moments
    
    if active_particles_indices is not None:
        return total_elc[active_particles_indices]
    return total_elc


def _compute_elcic_far_forces(system, params: dict, lambda_param: float):
    """
    PLACEHOLDER FOR CURSOR: Implement the analytical far formula forces here.
    This calculates forces on real particles (L_0) exerted by far images (L_+2, L_-2, etc.)
    
    Dielectric prefactors:
        delta_t = params['delta_mid_top']
        delta_b = params['delta_mid_bot']
    Slab boundaries are at z = 0 (bottom) and z = gap_size (top).
    
    Mathematical Structure factors to implement for far images:
        Implement the geometric progression sums over the infinite series of images
        using the exponential product decompositions detailed in the prompt equations.
    """
    lx, ly, lz = system.box_l
    gap_size = params['gap_size']
    delta_t = params['delta_mid_top']
    delta_b = params['delta_mid_bot']
    pw_err = params.get('pw_error', 1e-8)
    
    particles = system.part.all()
    n_part = len(particles)
    qs = particles.q
    xs, ys, zs = particles.pos.T
    
    f_far = np.zeros((n_part, 3))
    
    # --- TODO: CURSOR IMPLEMENTS THE FAR BLUEPRINT HERE ---
    
    return f_far


def get_elcic_forces(system, params: dict, lambda_param: float = 5.0):
    """
    Main ELCIC Force Orchestrator.
    Gradually extend this routine across the roadmap steps.
    """
    prefactor = params['prefactor']
    gap_size = params['gap_size']
    delta_t = params['delta_mid_top']
    delta_b = params['delta_mid_bot']
    
    # 1. TODO: Step 2+ Particle Classification
    # Group real particles into L_0,0, L_0,+1, L_0,-1 based on distance to boundaries (0 and gap_size)
    
    # 2. TODO: Step 3+ Construct the virtual Super-system L_T = L_-1 U L_0 U L_+1
    # Create a temporary simulation system context, populate primary image charges
    # with scaled charges (q * delta) and mirror positions.
    
    # 3. Compute Near Forces using existing ELC workflow on the expanded system context
    # (For Step 1, this just processes the default system)
    f_near = _get_f_3d(system, params) + prefactor * _get_elc_correction(system, params)
    
    # 4. TODO: Step 2+ Compute Analytical Far Forces
    f_far = _compute_elcic_far_forces(system, params, lambda_param)
    
    # Total combined force return vector
    return f_near + f_far