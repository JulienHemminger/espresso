import numpy as np
import espressomd
import espressomd.electrostatics


def get_elcic_forces(system, params: dict):
    """Computes electrostatic forces for 2D+h slab systems with dielectric interfaces

    using the ELCIC method with strict baseline handling.
    """
    box = np.array(system.box_l)
    lx, ly, lz_full = box[0], box[1], box[2]
    prefactor = params.get("prefactor", 1.0)
    gap_size = params["gap_size"]
    pw_err = params.get("pw_error", 1e-8)

    # Physical boundaries defining the slab height
    h = lz_full - gap_size

    delta_t = params.get("delta_mid_top", 0.0)
    delta_b = params.get("delta_mid_bot", 0.0)
    delta_prod = delta_t * delta_b
    lambda_ = params.get("lambda", 0.0)

    print("=" * 60)
    print(" DIAGNOSTIC ELCIC LOG: INITIALIZING BALANCED PASS")
    print("=" * 60)
    print(f"Geometry: lx={lx:.4f}, ly={ly:.4f}, lz_full={lz_full:.4f}")
    print(f"Slab Region (h): {h:.4f}, Gap: {gap_size:.4f}")
    print(f"Dielectrics: Delta_top={delta_t:.4f}, Delta_bot={delta_b:.4f}")
    print(f"Boundary Thickness (lambda): {lambda_:.4f}")

    # Capture original tracking configuration
    parts = system.part.all()
    n_real = len(parts)
    qs = np.array([p.q for p in parts])
    pos = np.array([p.pos for p in parts])
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]


    # --- Step 1: Direct 3D Periodic Reference Evaluation ---
    p3m_base = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_err,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m_base
    system.integrator.run(0)
    f_3d_baseline = np.array([p.f for p in parts])

    # --- Step 2: Analytical Reciprocal Layer Correction Term ---
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_range = np.arange(-p_max, p_max + 1)
    q_range = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (
        np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max
    )
    pk, qk = P[mask], Q[mask]
    fx, fy = pk / lx, qk / ly
    f_mag = np.sqrt(fx**2 + fy**2)

    arg_x = 2.0 * np.pi * fx
    arg_y = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f_mag

    # Trigonometric functions for phase decomposition
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    # Factorization matrices tracking coordinates relative to borders
    exp_plus = np.exp(arg_z * zs[:, None])
    exp_minus = np.exp(-arg_z * zs[:, None])

    # Component projections mirroring Eq 3.3 product decompositions
    chi_p = [
        np.sum(qs[:, None] * exp_plus * cx * cy, axis=0),
        np.sum(qs[:, None] * exp_plus * sx * cy, axis=0),
        np.sum(qs[:, None] * exp_plus * cx * sy, axis=0),
        np.sum(qs[:, None] * exp_plus * sx * sy, axis=0),
    ]
    chi_m = [
        np.sum(qs[:, None] * exp_minus * cx * cy, axis=0),
        np.sum(qs[:, None] * exp_minus * sx * cy, axis=0),
        np.sum(qs[:, None] * exp_minus * cx * sy, axis=0),
        np.sum(qs[:, None] * exp_minus * sx * sy, axis=0),
    ]

    # Geometric infinite progressions denominator tracking interface reflections
    denom_prog = 1.0 / (1.0 - delta_prod * np.exp(-2.0 * arg_z * h))
    term_pref = (1.0 / (lx * ly * f_mag)) * denom_prog

    f_elcic_recip = np.zeros((n_real, 3))
    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        # Analytical combination scaling the continuous mirror image layers
        exp_factor = np.exp(-2.0 * arg_z * h)
        combined_fields = (
            delta_t * exp_factor * exp_plus * chi_m[i]
            + delta_b * exp_factor * exp_minus * chi_p[i]
            + delta_prod * exp_factor * (exp_plus * chi_p[i] + exp_minus * chi_m[i])
        )

        f_elcic_recip[:, 0] += (
            qs[:, None] * dtx * ty * combined_fields @ term_pref
        )
        f_elcic_recip[:, 1] += (
            qs[:, None] * tx * dty * combined_fields @ term_pref
        )

        # Signed Z components matching normal-direction asymmetry reflections
        z_fields = (
            delta_t * exp_factor * exp_plus * chi_m[i]
            - delta_b * exp_factor * exp_minus * chi_p[i]
        )
        f_elcic_recip[:, 2] += (
            qs[:, None] * arg_z * tx * ty * z_fields @ term_pref
        )

    # --- Step 3: Complete Non-Neutral/Slab Background Alignment ---
    f_corr_moments = np.zeros((n_real, 3))
    
    # Fundamental global moments
    xi0 = np.sum(qs)        # Net charge of the real particles
    xi1 = np.sum(qs * zs)   # Net dipole moment along z
    
    # 1. Primary dielectric layer progression matching term
    if abs(1.0 - delta_prod) > 1e-9:
        pref_moments = -(4.0 * np.pi / (lx * ly * h)) * (1.0 / (1.0 - delta_prod))
        f_corr_moments[:, 2] = (
            pref_moments
            * qs
            * (xi1 * (1.0 + delta_prod) - xi0 * zs * (1.0 - delta_prod))
        )

    # 2. Complete 3D Periodic Box Background Subtraction (Force Derivative)
    # This accounts for BOTH the linear coordinate drift AND any non-neutral 
    # layer slicing constraints from the full box volume.
    volume_factor = lx * ly * lz_full
    
    # Standard ELC linear field correction
    f_corr_moments[:, 2] += (4.0 * np.pi / volume_factor) * qs * zs * xi0
    
    # Constant volume shift correction for segmented neutrality frames
    f_corr_moments[:, 2] -= (4.0 * np.pi / volume_factor) * qs * (xi1 - (lz_full / 2.0) * xi0)


    f_elcic_corr = prefactor * (f_elcic_recip + f_corr_moments)

    # --- Step 4: Near-field Layer Correction via Disjoint Set Evaluation ---
    f_near_field_images = np.zeros((n_real, 3))
    idx_bot = np.where((zs >= 0.0) & (zs <= lambda_))[0]
    idx_top = np.where((zs >= (h - lambda_)) & (zs <= h))[0]

    if len(idx_bot) > 0 or len(idx_top) > 0:
        # Save positions and charges before clearing
        saved_configuration = [(p.pos.copy(), p.q) for p in system.part.all()]
        system.part.clear()

        # Add original reference particles
        for p_idx in range(n_real):
            system.part.add(
                pos=saved_configuration[p_idx][0],
                q=saved_configuration[p_idx][1],
            )

        # Inject localized 1st-order image reflections
        for idx in idx_bot:
            system.part.add(pos=np.array([xs[idx], ys[idx], -zs[idx]]), q=delta_b * qs[idx])

        for idx in idx_top:
            system.part.add(
                pos=np.array([xs[idx], ys[idx], 2.0 * h - zs[idx]]),
                q=delta_t * qs[idx],
            )

        # Isolated local near-field evaluation step
        p3m_near = espressomd.electrostatics.P3M(
            prefactor=prefactor,
            accuracy=pw_err,
            check_neutrality=False,
            verbose=False,
        )
        system.electrostatics.solver = p3m_near
        system.integrator.run(0)

        parts_near = list(system.part.all())
        for idx in range(n_real):
            f_near_field_images[idx] = parts_near[idx].f - f_3d_baseline[idx]

        # Restore native configuration cleanly
        system.part.clear()
        for p_idx in range(n_real):
            system.part.add(
                pos=saved_configuration[p_idx][0],
                q=saved_configuration[p_idx][1],
            )
        system.electrostatics.solver = p3m_base

    # --- Step 5: Final Aggregation and Detailed Tracking Printouts ---
    f_elcic_corr = prefactor * (f_elcic_recip + f_corr_moments)
    f_total_elcic = f_3d_baseline + f_elcic_corr + f_near_field_images

    print("\n" + "="*60)
    print("         ELCIC COMPONENT-WISE TELEMETRY LOG")
    print("="*60)
    for idx in range(n_real):
        print(f"--- Particle {idx} (q={qs[idx]}, z={zs[idx]:.4f}) ---")
        print(f"  3D Baseline P3M F : {f_3d_baseline[idx]}")
        print(f"  Reciprocal ELCIC F: {prefactor * f_elcic_recip[idx]}")
        print(f"  Moment CorrectionF: {prefactor * f_corr_moments[idx]}")
        print(f"  Near-Field Image F: {f_near_field_images[idx]}")
        print(f"  Computed Total F  : {f_total_elcic[idx]}")
        
        # Replace this line with your analytical or high-accuracy reference grid truth vector
        f_truth_actual = np.array([0.0, 0.0, 0.0]) 
        
        if np.any(f_truth_actual):
            err = f_total_elcic[idx] - f_truth_actual
            print(f"  Absolute Error Vec: {err}")
            print(f"  Max Absolute Error: {np.max(np.abs(err)):.4e}")
    print("="*60 + "\n")

    return f_total_elcic