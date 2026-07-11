import espressomd
import espressomd.electrostatics
import numpy as np


def get_elc_forces_contribs(system, gap_size=1.0, pw_err=1e-6, prefactor=1.0):
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    charges = particles.q
    xs, ys, z_coordinates = particles.pos.T
    volume = lx * ly * lz

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_err, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)

    f_3D = np.array([p.f for p in particles])

    xi0 = np.sum(charges)
    xi1 = np.sum(charges * z_coordinates)

    f_dipole = np.zeros((n_part, 3))
    f_dipole[:, 2] = (
        -(4.0 * np.pi / (lx * ly * lz)) * charges * (xi1 - xi0 * z_coordinates)
    )

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
    ex_p, ex_m = (
        np.exp(arg_z * z_coordinates[:, None]),
        np.exp(-arg_z * z_coordinates[:, None]),
    )

    def get_chi(ez, tx, ty):
        return np.sum(charges[:, None] * ez * tx * ty, axis=0)

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

    f_far = np.zeros((n_part, 3))

    for i in range(4):
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy

        # Reciprocal X force
        f_far[:, 0] += (
            charges[:, None]
            * (ex_p * dtx * ty * chi_m[i] + ex_m * dtx * ty * chi_p[i])
            @ term_pref
        )

        # Reciprocal Y force
        f_far[:, 1] += (
            charges[:, None]
            * (ex_p * tx * dty * chi_m[i] + ex_m * tx * dty * chi_p[i])
            @ term_pref
        )

        # Reciprocal Z force
        f_far[:, 2] += (
            charges[:, None]
            * arg_z
            * (ex_p * tx * ty * chi_m[i] - ex_m * tx * ty * chi_p[i])
            @ term_pref
        )

    return (prefactor, f_3D, f_far, f_dipole)


def get_elc_forces(system, gap_size=1.0, pw_err=1e-6, prefactor=1.0):
    prefactor, f_3d, f_elc_recip, f_corr_moments = get_elc_forces_contribs(
        system, gap_size, pw_err, prefactor
    )

    # Total Force Assembly
    f_final = f_3d + prefactor * (f_elc_recip + f_corr_moments)
    return [f for f in f_final]
