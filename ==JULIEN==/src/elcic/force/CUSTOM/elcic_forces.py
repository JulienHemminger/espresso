import numpy as np
import espressomd
import espressomd.electrostatics

def get_elcic_forces(system, params: dict):
    gap_size = params['gap_size']
    prefactor = params['prefactor']
    pw_err = 1e-8
    
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    qs = particles.q
    xs, ys, zs = particles.pos.T
    volume = lx * ly * lz

    # 1. 3D Periodic Forces from P3M
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    f_3d = np.array([p.f for p in particles])

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
    f_final = f_3d + prefactor * (f_elc_recip + f_corr_moments)
    return [f for f in f_final]