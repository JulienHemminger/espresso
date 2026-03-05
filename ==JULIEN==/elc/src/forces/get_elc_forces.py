import numpy as np
import espressomd
import espressomd.electrostatics

def get_elc_forces(system, gap_size=1.0, pw_err=1e-6) -> list[np.ndarray]:
    lx, ly, lz = system.box_l
    particles = system.part.all()
    n_part = len(particles)
    qs = particles.q
    xs, ys, zs = particles.pos.T

    # 1. 3D Periodic Forces from P3M
    # We assume the system box already includes the gap_size in the Z dimension
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err, check_neutrality=False)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    # Extract 3D forces and prefactor
    f_total = np.array([p.f for p in particles])
    prefactor = p3m.prefactor

    # 2. Dipole Correction (Neutral System)
    # E_dipole = 2*pi/(lx*ly*lz) * (sum q_i z_i)^2
    xi1 = np.sum(qs * zs)
    dip_fac = 4.0 * np.pi / (lx * ly * lz)
    f_dipole = np.zeros((n_part, 3))
    f_dipole[:, 2] = -dip_fac * qs * xi1

    # 3. Reciprocal Space ELC Correction
    # Parameters for spectral truncation
    f_max = -np.log(pw_err) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_range = np.arange(-p_max, p_max + 1)
    q_range = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()

    # Exclude k=0 and apply circular truncation
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P/lx)**2 + (Q/ly)**2) <= f_max)
    pk, qk = P[mask], Q[mask]
    fx, fy = pk / lx, qk / ly
    f = np.sqrt(fx**2 + fy**2)
    
    arg_x = 2.0 * np.pi * fx
    arg_y = 2.0 * np.pi * fy
    arg_z = 2.0 * np.pi * f
    
    # Precompute trig and exp terms for all particles and k-vectors
    # Shapes: (n_part, n_k)
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Chi functions (Product decomposition) [cite: 67]
    def get_chi(ez, tx, ty):
        return np.sum(qs[:, None] * ez * tx * ty, axis=0)

    # Combinations for 2D Fourier
    chi_p = [get_chi(ex_p, cx, cy), get_chi(ex_p, sx, cy), 
             get_chi(ex_p, cx, sy), get_chi(ex_p, sx, sy)]
    chi_m = [get_chi(ex_m, cx, cy), get_chi(ex_m, sx, cy), 
             get_chi(ex_m, cx, sy), get_chi(ex_m, sx, sy)]

    # Weighting factor for the replicas [cite: 76]
    # rep = exp(-2*pi*f*lz) / (1 - exp(-2*pi*f*lz))
    # Note: Arnold 2002 uses 4*pi*L_z*f in denominator for certain conventions, 
    # but 2*pi*f*lz matches standard spectral layer ELC for slab replicas.
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    term_pref = (1.0 / (lx * ly * f)) * rep

    f_elc_recip = np.zeros((n_part, 3))

    # Gradient Calculation:
    # d/dx (sin(ax)) = a*cos(ax), d/dx (cos(ax)) = -a*sin(ax)
    # d/dz (exp(az)) = a*exp(az)
    for i in range(4):
        # Determine which trig functions to use based on index i
        tx = cx if i in [0, 2] else sx
        ty = cy if i in [0, 1] else sy
        dtx = -arg_x * sx if i in [0, 2] else arg_x * cx
        dty = -arg_y * sy if i in [0, 1] else arg_y * cy
        
        # x-force: - dE/dx
        # Contribution: q_i * (exp_p * dtx * ty * chi_m + exp_m * dtx * ty * chi_p)
        f_elc_recip[:, 0] += qs[:, None] * (ex_p * dtx * ty * chi_m[i] + 
                                           ex_m * dtx * ty * chi_p[i]) @ term_pref
        
        # y-force: - dE/dy
        f_elc_recip[:, 1] += qs[:, None] * (ex_p * tx * dty * chi_m[i] + 
                                           ex_m * tx * dty * chi_p[i]) @ term_pref
        
        # z-force: - dE/dz
        # d/dz (ex_p) = arg_z * ex_p, d/dz (ex_m) = -arg_z * ex_m
        f_elc_recip[:, 2] += qs[:, None] * arg_z * (ex_p * tx * ty * chi_m[i] - 
                                                   ex_m * tx * ty * chi_p[i]) @ term_pref

    # Total ELC Force = 3D Force + Prefactor * (Dipole Correction + Reciprocal Correction)
    # Note the sign: E_total = E_3D + E_corr => F_total = F_3D - grad(E_corr)
    # The gradients above were already computed as -grad(E).
    f_final = f_total + prefactor * (f_dipole + f_elc_recip)

    return [f for f in f_final]