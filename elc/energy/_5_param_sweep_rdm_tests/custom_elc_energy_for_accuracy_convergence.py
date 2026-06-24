import espressomd
import espressomd.electrostatics
import numpy as np


def get_elc_energy_contribs(gap_size, pw_error, system, prefactor=1.0):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    # 1. 3D Periodic Energy from P3M
    system.electrostatics.solver = p3m
    E_3d = system.analysis.energy()["total"]


    lx, ly, lz = system.box_l
    particles = system.part.all()
    qs, (xs, ys, zs) = particles.q, particles.pos.T

    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)
    volume = lx * ly * lz
    E_non_neutral_corr = 2.0 * np.pi / volume * (- xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    E_dipole = 2.0 * np.pi / volume * xi1**2
    

    # 4. Reciprocal Space ELC Term
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    # Exclude the k=0 mode (handled by the real space and dipole terms)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    # Particle-wise components
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Compute form factors (Chi) linearly
    def s_term(ez, c1, c2):
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Summing over the four combinations of sin/cos for the 2D Fourier transform
    chi = (
        s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy)
        + s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy)
        + s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy)
        + s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy)
    )

    # The reciprocal energy correction
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    E_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)

    return (float(prefactor), float(E_recip), float(E_3d), float(E_dipole + E_non_neutral_corr))


def get_elc_energy(system, gap_size, pw_error, prefactor=1.0):
    _, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(
        gap_size, pw_error, system, prefactor
    )
    return e_3d + (prefactor * e_non_neutral_corr) + (prefactor * e_recip)
