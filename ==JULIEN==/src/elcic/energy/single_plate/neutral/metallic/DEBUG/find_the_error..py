import espressomd
import espressomd.electrostatics
import numpy as np

def get_legacy_energy(system, params_dict):
    system.electrostatics.clear()
    gap_size      = params_dict["gap_size"]
    pw_error      = params_dict["pw_error"]
    prefactor     = params_dict.get("prefactor", 1.0)
    delta_mid_top = params_dict.get("delta_mid_top")
    delta_mid_bot = params_dict.get("delta_mid_bot")
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=True,
    )

    args = {
        "actor": p3m,
        "gap_size": gap_size,
        "maxPWerror": pw_error,
        "check_neutrality": False,
        "neutralize": False,
    }

    if delta_mid_top is not None:
        args["delta_mid_top"] = delta_mid_top
    if delta_mid_bot is not None:
        args["delta_mid_bot"] = delta_mid_bot

    elc_legacy = espressomd.electrostatics.ELC(**args)

    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    energy = system.analysis.energy()["total"]
    system.electrostatics.clear()
    return energy
    

def _get_non_neutral_correction(box, zs, qs):
    """Equation 3.10 & 3.11: Dipole and non-neutrality energy correction."""
    lx, ly, lz = box
    xi0, xi1, xi2 = np.sum(qs), np.sum(qs * zs), np.sum(qs * zs**2)
    fac = 2.0 * np.pi / (lx * ly * lz)
    return fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

def _get_chi_components(fx, fy, f, pos, qs, sign=1):
    """Equation 3.3: Product decomposition for the ELC reciprocal sum."""
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    xs, ys, zs = pos.T
    
    if sign != 0:
        ez = np.exp(sign * 1j * arg_z * zs[:, None])
    else:
        ez = 1.0
    
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])

    # Returns [CC, SC, CS, SS]
    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0) 
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    """Evaluates Equations 4.6 through 4.13 for far image interactions."""
    lx, ly, lz = box
    delta = db * dt
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)

    # Generate reciprocal space grid
    p_range = np.arange(-np.ceil(f_max * lx), np.ceil(f_max * lx) + 1)
    q_range = np.arange(-np.ceil(f_max * ly), np.ceil(f_max * ly) + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    # Pre-calculate base Chi factors with imaginary exponentials
    chi0_p = _get_chi_components(fx, fy, f, ps, qs, sign=1)
    chi0_m = _get_chi_components(fx, fy, f, ps, qs, sign=-1)

    def l_pq_sum(z_dist, delta_coeff):
        """Proper geometric series sum for L_pq."""
        exp_term = np.exp(-2.0 * np.pi * f * z_dist)
        denom = 1.0 - delta * np.exp(-4.0 * np.pi * f * lz)
        return delta_coeff * exp_term / denom

    def compute_field_contribution(mask, d_coeff, z_transform):
        if not np.any(mask):
            return [np.zeros_like(f) for _ in range(4)]
        
        chi_local = _get_chi_components(fx, fy, f, ps[mask], qs[mask], sign=0)
        z_transformed = z_transform(ps[mask, 2, None])
        
        # For particles in the gap region
        term1 = l_pq_sum(2 * lz + z_transformed, d_coeff)
        term2 = l_pq_sum(2 * lz - z_transformed, delta)
        
        combined_term = np.sum(term1 + term2, axis=0)
        return [c * combined_term for c in chi_local]

    # Bottom (L-2) and Top (L+2) far field logic
    m_bot = ps[:, 2] <= gap_size
    m_top = ps[:, 2] > (lz - gap_size)
    
    # Chi_-2^+ components (Eq 4.8 and 4.9)
    chi_m2_p_bot = compute_field_contribution(m_bot, db, lambda z: z)
    chi_m2_p_top = compute_field_contribution(~m_bot, 1.0, lambda z: -2*lz + z)
    chi_m2_p = [a + b for a, b in zip(chi_m2_p_bot, chi_m2_p_top)]
    
    # Chi_+2^- components (Eq 4.10 and 4.11)
    chi_p2_m_top = compute_field_contribution(m_top, dt, lambda z: lz - z)
    chi_p2_m_bot = compute_field_contribution(~m_top, 1.0, lambda z: -lz - z)
    chi_p2_m = [a + b for a, b in zip(chi_p2_m_top, chi_p2_m_bot)]

    # Final Energy assembly (Eq 4.6 and 4.7)
    pref = 1.0 / (lx * ly)
    
    # Real part extraction for energy
    e_far = 0.0
    for chi_m, chi_p in zip(chi0_m, chi_m2_p):
        e_far += pref * np.sum((1.0 / f) * np.real(np.conj(chi_m) * chi_p))
    
    for chi_p, chi_m in zip(chi0_p, chi_p2_m):
        e_far += pref * np.sum((1.0 / f) * np.real(np.conj(chi_p) * chi_m))
    
    return e_far

def _get_config_energy(system, p_set, q_set, prefactor, accuracy):
    """Helper to swap system particles and compute P3M + Correction."""
    system.part.clear()
    system.part.add(pos=p_set, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    e_3d = system.analysis.energy()["total"]
    system.electrostatics.clear()
    e_corr = prefactor * _get_non_neutral_correction(system.box_l, p_set[:, 2], q_set)
    return e_3d + e_corr

def get_custom_energy(system, params: dict):
    """
    Main ELCIC energy calculation.
    """
    box = system.box_l
    lz = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]

    # Cache original state
    parts = system.part.all()
    qs_orig, ps_orig = parts.q.copy(), parts.pos.copy()

    # 1. Generate Near-Image Coordinates
    m_bot = ps_orig[:, 2] <= gap
    m_top = ps_orig[:, 2] > (lz - gap)
    
    ps_m1 = ps_orig[m_bot].copy()
    ps_m1[:, 2] *= -1
    
    ps_p1 = ps_orig[m_top].copy()
    ps_p1[:, 2] = 2 * lz - ps_p1[:, 2]

    qs_m1 = qs_orig[m_bot] * db
    qs_p1 = qs_orig[m_top] * dt

    # 2. Linear Combination for Near-Field (Eq. 4.14)
    e_l0 = _get_config_energy(system, ps_orig, qs_orig, pref, eps)
    
    if len(qs_m1) > 0 or len(qs_p1) > 0:
        e_pm1 = _get_config_energy(system, 
                                   np.vstack([ps_m1, ps_p1]) if len(ps_m1) > 0 and len(ps_p1) > 0 
                                   else (ps_m1 if len(ps_m1) > 0 else ps_p1),
                                   np.concatenate([qs_m1, qs_p1]), pref, eps)
        e_lt = _get_config_energy(system, 
                                  np.vstack([ps_orig, ps_m1, ps_p1]) if len(ps_m1) > 0 and len(ps_p1) > 0
                                  else (np.vstack([ps_orig, ps_m1]) if len(ps_m1) > 0 else np.vstack([ps_orig, ps_p1])),
                                  np.concatenate([qs_orig, qs_m1, qs_p1]), pref, eps)
        e_near = 0.5 * (e_lt - e_pm1 + e_l0)
    else:
        e_near = e_l0

    # 3. Far-Field Correction
    e_far = pref * _get_far_field_energy(box, gap, eps, qs_orig, ps_orig, db, dt)

    # Restore original state
    system.part.clear()
    system.part.add(pos=ps_orig, q=qs_orig)

    return e_near + e_far

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4


def run_elc(params):
    system.electrostatics.clear()
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    legacy_energy = get_legacy_energy(system, params)


    analytical_energy = -0.18116638064555976
    custom_energy = float(get_custom_energy(system, params))

    print(f"{analytical_energy=}")
    print(f"{legacy_energy=}, error={legacy_energy-analytical_energy}")
    print(f"{custom_energy=}, error={custom_energy-analytical_energy}")



params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
params["lz"] = params["gap_size"] + 10

run_elc(params)
