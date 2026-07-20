import espressomd
import espressomd.electrostatics
import numpy as np

from src.elc.force.legacy_elc_forces import get_legacy_forces
from src.elcic.force.CUSTOM.elcic_forces import get_elcic_forces


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
    f_3d = np.array([p.f for p in parts])

    # --- Step 2: Analytical Reciprocal Layer Correction Term ---
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

    f_far = np.zeros((n_real, 3))
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

        f_far[:, 0] += qs[:, None] * dtx * ty * combined_fields @ term_pref
        f_far[:, 1] += qs[:, None] * tx * dty * combined_fields @ term_pref

        # Signed Z components matching normal-direction asymmetry reflections
        z_fields = (
            delta_t * exp_factor * exp_plus * chi_m[i]
            - delta_b * exp_factor * exp_minus * chi_p[i]
        )
        f_far[:, 2] += qs[:, None] * arg_z * tx * ty * z_fields @ term_pref

    # --- Step 3: Complete Non-Neutral/Slab Background Alignment ---
    # --- Step 3: Corrected Non-Neutral/Slab Background Alignment ---
    f_dipole = np.zeros((n_real, 3))

    # Fundamental global moments
    xi0 = np.sum(qs)  # Net charge of the system
    xi1 = np.sum(qs * zs)  # Net dipole moment along z

    # 1. Primary dielectric layer progression matching term
    if abs(1.0 - delta_prod) > 1e-9:
        pref_moments = -(4.0 * np.pi / (lx * ly * h)) * (1.0 / (1.0 - delta_prod))
        f_dipole[:, 2] = (
            pref_moments
            * qs
            * (xi1 * (1.0 + delta_prod) - xi0 * zs * (1.0 - delta_prod))
        )

    # 2. Volume-shift adjustments
    # These terms must scale proportionally to net-charge asymmetry (xi0).
    # For charge-neutral systems (xi0 = 0), these background forces must be zero.
    volume_factor = lx * ly * lz_full

    # Linear and constant alignment updates
    f_dipole[:, 2] += (4.0 * np.pi / volume_factor) * qs * zs * xi0
    f_dipole[:, 2] -= (4.0 * np.pi / volume_factor) * qs * (xi1 - (lz_full / 2.0)) * xi0

    # --- Step 4: Near-field Layer Correction via Disjoint Set Evaluation ---
    f_idk = np.zeros((n_real, 3))
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
            system.part.add(
                pos=np.array([xs[idx], ys[idx], -zs[idx]]), q=delta_b * qs[idx]
            )

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
            f_idk[idx] = parts_near[idx].f - f_3d[idx]

        # Restore native configuration cleanly
        system.part.clear()
        for p_idx in range(n_real):
            system.part.add(
                pos=saved_configuration[p_idx][0],
                q=saved_configuration[p_idx][1],
            )
        system.electrostatics.solver = p3m_base

    # --- Step 5: Final Aggregation and Detailed Tracking Printouts ---
    f_total_elcic = f_3d + f_far + f_dipole

    print("\n" + "=" * 60)
    print("         ELCIC COMPONENT-WISE TELEMETRY LOG")
    print("=" * 60)
    for idx in range(n_real):
        print(f"--- Particle {idx} (q={qs[idx]}, z={zs[idx]:.4f}) ---")
        print(f"  3D Baseline P3M F : {f_3d[idx]}")
        print(f"  Reciprocal ELCIC F: {prefactor * f_far[idx]}")
        print(f"  Moment CorrectionF: {prefactor * f_dipole[idx]}")
        print(f"  Near-Field Image F: {f_idk[idx]}")
        print(f"  Computed Total F  : {f_total_elcic[idx]}")

        # Replace this line with your analytical or high-accuracy reference grid truth vector
        f_truth_actual = np.array([0.0, 0.0, 0.0])

        if np.any(f_truth_actual):
            err = f_total_elcic[idx] - f_truth_actual
            print(f"  Absolute Error Vec: {err}")
            print(f"  Max Absolute Error: {np.max(np.abs(err)):.4e}")
    print("=" * 60 + "\n")

    return {
        "F_total": f_total_elcic,
        "F_3D": f_3d,
        "F_dipole": f_dipole,
        "F_far": f_far,
    }


system = espressomd.System(box_l=[10, 10, 10])
system.time_step = 0.01
system.cell_system.skin = 0.4


params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "lambda": 1,
    "positions": [np.array([4, 5, 6]), np.array([1, 2, 3])],
}
params["lz"] = params["gap_size"] + 10


system.part.clear()
system.box_l = [params["lx"], params["ly"], params["lz"]]
for i in range(len(params["charges"])):
    system.part.add(pos=params["positions"][i], q=params["charges"][i])

for pw_error in [1e-4]:  # do a loop over all requested accuracies (pw_error)
    params["pw_error"] = pw_error
    custom_forces = get_elcic_forces(system, params)  # dict
    """
    e.g.
    {'F_total': array([[-0.0124852 , -0.01248282, -0.03247063], [ 0.01249053,  0.01248816,  0.03247779]]),
       
       'F_3D': array([[-0.01237177, -0.01236939, -0.03247421],
       [ 0.01237177,  0.01236939,  0.03247421]]),
       
       'F_dipole': array([[0., 0., 0.],
       [0., 0., 0.]]),
       
       'F_far': array([[-1.13428333e-04, -1.13428333e-04,  3.58080998e-06],
       [ 1.18763070e-04,  1.18763070e-04,  3.58080998e-06]])}
    """

    legacy_forces = get_legacy_forces(system, params)  # list
    # e.g. [array([-0.01224997, -0.01224997, -0.05344835]), array([0.01225465, 0.01225465, 0.03427915])]

"""



pick particle 1
plot contribs and force magnitude
"""
