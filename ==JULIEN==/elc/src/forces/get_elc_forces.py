import numpy as np
import espressomd
import espressomd.electrostatics

def get_elc_forces(system, gap_size=1.0, pw_err=1e-6) -> list[np.ndarray]:
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q
    xs, ys, zs = parts.pos.T
    n_part = len(qs)

    # 1. Base 3D P3M Forces
    # Note: Accuracy should be high enough to handle the gap
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    f_3d = np.array([p.f for p in parts])
    pref = p3m.prefactor

    # 2. Dipole Correction (z-direction only for neutral systems)
    volume = lx * ly * lz
    xi1 = np.sum(qs * zs)
    f_dip = np.zeros((n_part, 3))
    f_dip[:, 2] = - (4.0 * np.pi / volume) * qs * xi1

    # 3. Reciprocal ELC Correction
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    # Generate k-vectors
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P/lx)**2 + (Q/ly)**2) <= f_max)
    kx, ky = 2.0 * np.pi * P[mask] / lx, 2.0 * np.pi * Q[mask] / ly
    k = np.sqrt(kx**2 + ky**2)

    # Precompute trigonometric and exponential terms
    # Shape: (n_part, n_kvectors)
    phase = kx * xs[:, None] + ky * ys[:, None]
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    exp_plus, exp_minus = np.exp(k * zs[:, None]), np.exp(-k * zs[:, None])

    # Compute Global Form Factors (sum over particles)
    # These are O(N) to compute
    A_p = np.sum(qs[:, None] * exp_plus * cos_p, axis=0)
    B_p = np.sum(qs[:, None] * exp_plus * sin_p, axis=0)
    A_m = np.sum(qs[:, None] * exp_minus * cos_p, axis=0)
    B_m = np.sum(qs[:, None] * exp_minus * sin_p, axis=0)

    # Spectral Factor (accounts for the infinite sum of z-images)
    # factor = 1 / (exp(k*Lz) - 1)
    spec_fac = 1.0 / (np.exp(k * lz) - 1.0)
    term_pref = (2.0 / (lx * ly * k)) * spec_fac

    # Compute Forces (O(N) per k-vector)
    # Derivatives of the energy term w.r.t xi, yi, zi
    f_recip = np.zeros((n_part, 3))

    # We use the fact that the reciprocal energy is proportional to:
    # Sum_k term_pref * [ (A_p*A_m + B_p*B_m) ]
    
    # Z-force: d/dzi
    # d/dzi (exp(k*zi)) = k*exp(k*zi) ; d/dzi (exp(-k*zi)) = -k*exp(-k*zi)
    f_recip[:, 2] = np.sum(term_pref * k * qs[:, None] * (
        exp_plus * (A_m * cos_p + B_m * sin_p) - 
        exp_minus * (A_p * cos_p + B_p * sin_p)
    ), axis=1)

    # X-force: d/dxi
    # d/dxi (cos(kx*xi + ky*yi)) = -kx*sin(...)
    f_recip[:, 0] = np.sum(term_pref * kx * qs[:, None] * (
        exp_plus * (A_m * (-sin_p) + B_m * cos_p) + 
        exp_minus * (A_p * (-sin_p) + B_p * cos_p)
    ), axis=1)

    # Y-force: d/dyi
    f_recip[:, 1] = np.sum(term_pref * ky * qs[:, None] * (
        exp_plus * (A_m * (-sin_p) + B_m * cos_p) + 
        exp_minus * (A_p * (-sin_p) + B_p * cos_p)
    ), axis=1)

    # Combine all contributions
    total_forces = f_3d + pref * (f_dip + f_recip)
    
    return [f for f in total_forces]