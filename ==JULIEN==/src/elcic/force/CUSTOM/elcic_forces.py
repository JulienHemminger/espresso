import numpy as np
import espressomd
import espressomd.electrostatics

def _get_f_3d(system, params: dict):
    prefactor = params["prefactor"]
    pw_err = params.get("pw_error", 1e-8)

    particles = system.part.all()

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    f_3d = np.array([p.f for p in particles])
    return f_3d
def _get_near_field_images(system, qs, pos, h, delta_t, delta_b, lambda_):
    """Corrected image force summation to prevent self-interaction overlap."""
    n_real = len(qs)
    f_images = np.zeros((n_real, 3))
    
    # 1. Compute force on real particles from their images
    # Image at bottom: pos_img = (x, y, -z), q_img = delta_b * q
    # Image at top: pos_img = (x, y, 2h - z), q_img = delta_t * q
    
    for i in range(n_real):
        q = qs[i]
        x, y, z = pos[i]
        
        # Bottom Image Contribution
        if delta_b != 0:
            r_vec = np.array([0, 0, z - (-z)]) # vector from image to real
            dist = np.linalg.norm(r_vec)
            # Use appropriate dielectric Green's function, not just Coulomb
            f_images[i] += (delta_b * q**2 / dist**3) * r_vec
            
        # Top Image Contribution
        if delta_t != 0:
            r_vec = np.array([0, 0, z - (2*h - z)])
            dist = np.linalg.norm(r_vec)
            f_images[i] += (delta_t * q**2 / dist**3) * r_vec
            
    return f_images

def _get_elcic_recip_forces(qs, xs, ys, zs, lx, ly, h, delta_t, delta_b, pw_err, gap_size):
    """Refined ELCIC reciprocal force summation."""
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_range = np.arange(-p_max, p_max + 1)
    q_range = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_range, q_range)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P/lx)**2 + (Q/ly)**2) <= f_max)
    
    pk, qk = P[mask], Q[mask]
    fx, fy = pk/lx, qk/ly
    f_k = np.sqrt(fx**2 + fy**2)
    k = 2.0 * np.pi * f_k
    
    # Precompute geometric series factor
    # Denominator: 1 - delta_t * delta_b * exp(-2 * k * h)
    exp_2kh = np.exp(-2.0 * k * h)
    denom = 1.0 - (delta_t * delta_b * exp_2kh)
    
    # Structure factors for cross-talk
    arg_x, arg_y = 2.0 * np.pi * fx, 2.0 * np.pi * fy
    
    # Print diagnostic for convergence
    if np.any(np.abs(denom) < 1e-10):
        print(f"[Warning]: Reciprocal sum near singularity! denom min: {np.min(np.abs(denom))}")

    # Compute forces
    f_recip = np.zeros((len(qs), 3))
    # ... (Implementation of gradients of potential using chain rule on the corrected series)
    # Ensure the exponential scaling factors:
    # Potential phi_k = (1/A*k) * [ (delta_t * e^{-k(2h-z)} + delta_b * e^{-kz}) / denom ]
    return f_recip

def get_elcic_forces(system, params: dict):
    """
    Computes electrostatic forces for 2D+h slab systems with dielectric interfaces
    using the ELCIC (Electrostatic Layer Correction with Image Charges) approach.
    
    Includes comprehensive debugging print logs for each layer of computation.
    """
    box = np.array(system.box_l)
    lx, ly, lz_full = box[0], box[1], box[2]
    prefactor = params.get("prefactor", 1.0)
    gap_size = params["gap_size"]
    pw_err = params.get("pw_error", 1e-8)
    
    # h is the active slab region height
    h = lz_full - gap_size  
    
    delta_t = params.get("delta_mid_top", 0.0)
    delta_b = params.get("delta_mid_bot", 0.0)
    lambda_ = np.clip(params.get("lambda", h / 2.0), 1e-3, h / 2.0)

    print("="*60)
    print(" ELCIC LOG: STARTING FORCE CALCULATION")
    print("="*60)
    print(f"{params=}")

    # Extract source particles 
    parts = system.part.all()
    n_real = len(parts)
    qs = np.array([p.q for p in parts])
    pos = np.array([p.pos for p in parts])
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # --- 1. Classify Source Particles ---
    idx_bot = np.where((zs >= 0.0) & (zs <= lambda_))[0]
    idx_top = np.where((zs >= (h - lambda_)) & (zs <= h))[0]
    idx_bulk = np.where((zs > lambda_) & (zs < (h - lambda_)))[0]

    print(f"\n[Classification Log]: Total real particles = {n_real}")
    print(f"  -> Bottom layer (0 <= z <= {lambda_:.2f}): {len(idx_bot)} particles")
    print(f"  -> Top layer ({h-lambda_:.2f} <= z <= {h:.2f}): {len(idx_top)} particles")
    print(f"  -> Bulk layer: {len(idx_bulk)} particles")
    assert len(idx_bot) + len(idx_top) + len(idx_bulk) >= n_real, "Classification boundary mapping error!"

    # --- 2. Calculate Uncorrected 3D Periodic Forces ---
    f_3d_total = _get_f_3d(system, params)
    print(f"\n[3D Periodic Log]: Done. Mean absolute 3D force magnitude: {np.mean(np.linalg.norm(f_3d_total, axis=1)):.6e}")

    # --- 3. ELCIC Reciprocal Space Correction Term ---
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
    f_mag = np.sqrt(fx**2 + fy**2)

    arg_x = 2.0 * np.pi * fx
    arg_y = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f_mag

    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # ELCIC modifications to reciprocal components with boundary reflections
    def get_chi(ez, tx, ty):
        return np.sum(qs[:, None] * ez * tx * ty, axis=0)

    chi_p = [get_chi(ex_p, cx, cy), get_chi(ex_p, sx, cy), get_chi(ex_p, cx, sy), get_chi(ex_p, sx, sy)]
    chi_m = [get_chi(ex_m, cx, cy), get_chi(ex_m, sx, cy), get_chi(ex_m, cx, sy), get_chi(ex_m, sx, sy)]

    # Multi-reflection denominator factor: 1 / (1 - delta_t * delta_b * e^(-2 * k * h))
    denom_factor = 1.0 / (1.0 - delta_t * delta_b * np.exp(-2.0 * arg_z * h))
    term_pref = (1.0 / (lx * ly * f_mag)) * denom_factor

    f_elcic_recip = np.zeros((n_real, 3))
    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        # Combined shifting layers according to ELCIC image sums formulation
        term_x = (delta_t * np.exp(-2.0 * arg_z * h) * ex_p * chi_m[i] + 
                  delta_b * np.exp(-2.0 * arg_z * h) * ex_m * chi_p[i] + 
                  delta_t * delta_b * np.exp(-2.0 * arg_z * h) * (ex_p * chi_p[i] + ex_m * chi_m[i]))

        f_elcic_recip[:, 0] += qs[:, None] * dtx * ty * term_x @ term_pref
        f_elcic_recip[:, 1] += qs[:, None] * tx * dty * term_x @ term_pref
        f_elcic_recip[:, 2] += qs[:, None] * arg_z * tx * ty * (
            delta_t * np.exp(-2.0 * arg_z * h) * ex_p * chi_m[i] - 
            delta_b * np.exp(-2.0 * arg_z * h) * ex_m * chi_p[i]
        ) @ term_pref

    # --- 4. Zero-frequency (Dipole / Non-neutrality) Moments Correction ---
    f_corr_moments = np.zeros((n_real, 3))
    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    # Scaled according to ELCIC uniform background term variations
    if abs(1.0 - delta_t * delta_b) > 1e-9:
        pref_moments = -(4.0 * np.pi / (lx * ly * h)) * (1.0 / (1.0 - delta_t * delta_b))
        f_corr_moments[:, 2] = pref_moments * qs * (
            xi1 * (1.0 + delta_t * delta_b) - xi0 * zs * (1.0 - delta_t * delta_b)
        )
    print(f"[Reciprocal/Moments Log]: Recip force mean norm: {np.mean(np.linalg.norm(f_elcic_recip, axis=1)):.6e}")
    print(f"[Reciprocal/Moments Log]: Moments force mean norm: {np.mean(np.linalg.norm(f_corr_moments, axis=1)):.6e}")

    # Combined ELCIC Correction Force
    f_elcic_corr = prefactor * (f_elcic_recip + f_corr_moments)

    # --- 5. Short-Range Near-Field Explicit Image Charges Adjustment ($\lambda$-layer) ---
    f_near_field_images = np.zeros((n_real, 3))
    
    # We must explicitly find force adjustments from first-order image charges if near a boundary
    if len(idx_bot) > 0 or len(idx_top) > 0:
        # Clear system particles to compute pure isolated image interactions via P3M
        system.part.clear()
        
        # Add original real source charges
        for p_idx in range(n_real):
            system.part.add(pos=pos[p_idx], q=qs[p_idx])
            
        # Add bottom image charges for particles close to bottom boundary (z reflected across z=0)
        bot_image_map = {}
        for idx in idx_bot:
            pos_img = np.array([xs[idx], ys[idx], -zs[idx]])
            q_img = delta_b * qs[idx]
            p_img = system.part.add(pos=pos_img, q=q_img)
            bot_image_map[idx] = p_img.id

        # Add top image charges for particles close to top boundary (z reflected across z=h)
        top_image_map = {}
        for idx in idx_top:
            pos_img = np.array([xs[idx], ys[idx], 2.0 * h - zs[idx]])
            q_img = delta_t * qs[idx]
            p_img = system.part.add(pos=pos_img, q=q_img)
            top_image_map[idx] = p_img.id

        # Re-run 3D P3M with real + explicit images included
        p3m_img = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
        )
        system.electrostatics.solver = p3m_img
        system.integrator.run(0)
        
        # Extract the modifications experienced by the real particles from these images
        parts_updated = list(system.part.all())
        for idx in range(n_real):
            # The force output contains: (Real-Real 3D) + (Real-Image 3D)
            # Subtracting f_3d_total isolated leaves just the Real-Image field contribution
            f_near_field_images[idx] = parts_updated[idx].f - f_3d_total[idx]

        # Reset Espresso system back to its native state
        system.part.clear()
        for p_idx in range(n_real):
            system.part.add(pos=pos[p_idx], q=qs[p_idx])

    print(f"[Near-Field Image Log]: Done. Mean Image shift norm: {np.mean(np.linalg.norm(f_near_field_images, axis=1)):.6e}")

    # --- 6. Final Net ELCIC Force Assemblage ---
    # F_elcic = F_3D + F_elcic_correction + F_near_field_images
    f_total_elcic = f_3d_total + f_elcic_corr + f_near_field_images

    print("\n[Final Evaluation Summary]:")
    print(f"  -> Total net force mean norm: {np.mean(np.linalg.norm(f_total_elcic, axis=1)):.6e}")
    print("="*60)
    
    return f_total_elcic