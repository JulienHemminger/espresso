import numpy as np
import espressomd
import espressomd.electrostatics


def _get_f_3d(system, params: dict):
    prefactor = params["prefactor"]
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
    gap_size = params["gap_size"]
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

    chi_p = [
        get_chi(ex_p, cx, cy),
        get_chi(ex_p, sx, cy),
        get_chi(ex_p, cx, sy),
        get_chi(ex_p, sx, sy),
    ]
    chi_m = [
        get_chi(ex_m, cx, cy),
        get_chi(ex_m, sx, cy),
        get_chi(ex_m, cx, sy),
        get_chi(ex_m, sx, sy),
    ]

    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    term_pref = (1.0 / (lx * ly * f)) * rep

    f_elc_recip = np.zeros((n_part, 3))
    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        f_elc_recip[:, 0] += (
            qs[:, None]
            * (ex_p * dtx * ty * chi_m[i] + ex_m * dtx * ty * chi_p[i])
            @ term_pref
        )
        f_elc_recip[:, 1] += (
            qs[:, None]
            * (ex_p * tx * dty * chi_m[i] + ex_m * tx * dty * chi_p[i])
            @ term_pref
        )
        f_elc_recip[:, 2] += (
            qs[:, None]
            * arg_z
            * (ex_p * tx * ty * chi_m[i] - ex_m * tx * ty * chi_p[i])
            @ term_pref
        )

    return f_elc_recip + f_corr_moments


def get_elcic_forces(system, params: dict):
    """
    Example Paramters:
    params = {
        "lx": 50.0,
        "ly": 50.0,
        "gap_size": 20.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": 0.0,
        "charges": [+1.0, -1.0],
        "pw_error": 1e-8,
        'lambda': 20,
        "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    }
    params["lz"] = params["gap_size"] + 40


    espressomd.System is a singleton, you can only have one instance
        to initialize a system:
            system = espressomd.System(box_l=[1, 2, 3])
            system.time_step = 0.01

        system.part.clear() # remove all particles

        system.part.add(pos=np.array([1, 2, 3]), q=1) # to add a particle

        system.box_l = [lx, ly, lz] # to resize a system, system needs to have no particles before doing this.

        to use P3M:
            p3m = espressomd.electrostatics.P3M(
                    prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
                )
            system.electrostatics.solver = p3m
            system.integrator.run(0)#
            energy = system.analysis.energy()["total"]

    """
    box = np.array(system.box_l)
    lz_full = box[2]
    prefactor = params.get("prefactor", 1.0)
    gap_size = params["gap_size"]
    lz = lz_full - gap_size  # Top interface position (gap is between z=0 and z=lz)


    parts = system.part.all()
    qs, ps = parts.q.copy(), parts.pos.copy()
    n_real = len(parts)
    zs = np.array([p.pos[2] for p in parts])

    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)
    
    idx_bot = np.where((zs >= 0.0) & (zs < lambda_))[0]
    idx_top = np.where((zs > (lz - lambda_)) & (zs <= lz))[0]
    idx_bulk = np.where((zs >= lambda_) & (zs <= (lz - lambda_)))[0]

    total_classified = len(idx_bot) + len(idx_top) + len(idx_bulk)
    assert total_classified == n_real, (
        f"Particle classification mismatch! {n_real=}, {total_classified=}"
    )

    f_3d_total = _get_f_3d(system, params)

    f_elc_total = _get_elc_correction(system, params)

    f_near = f_3d_total + prefactor * f_elc_total

    return f_near
