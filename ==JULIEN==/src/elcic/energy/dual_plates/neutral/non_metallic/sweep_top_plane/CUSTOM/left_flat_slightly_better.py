import espressomd
import espressomd.electrostatics
import numpy as np

def _get_chi_components(fx, fy, f, pos, qs, z_offset):
    """Computes z-dependent chi components for the ELC reciprocal sum."""
    arg_x, arg_y = 2.0 * np.pi * fx, 2.0 * np.pi * fy
    xs, ys, zs = pos.T
    
    # We apply the z-shift to calculate the interaction with the image plane
    ez = np.exp(-2.0 * np.pi * f * np.abs(zs[:, None] - z_offset))
    
    cx = np.cos(arg_x * xs[:, None])
    sx = np.sin(arg_x * xs[:, None])
    cy = np.cos(arg_y * ys[:, None])
    sy = np.sin(arg_y * ys[:, None])

    return [np.sum(qs[:, None] * ez * c1 * c2, axis=0) 
            for c1, c2 in [(cx, cy), (sx, cy), (cx, sy), (sx, sy)]]

def _get_far_field_energy(box, gap_size, pw_error, qs, ps, dt):
    """
    Computes the 2D-periodic reciprocal correction (Far-Field).
    """
    lx, ly = box[0], box[1]
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    
    # Generate reciprocal vectors
    k = np.ceil(f_max * max(lx, ly))
    p_range = np.arange(-k, k + 1)
    q_range = np.arange(-k, k + 1)
    P, Q = np.meshgrid(p_range, q_range)
    P, Q = P.flatten(), Q.flatten()
    
    mask = (P != 0) | (Q != 0)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)
    f_mask = f <= f_max
    fx, fy, f = fx[f_mask], fy[f_mask], f[f_mask]

    # Interface reflection plane
    z_int = box[2] - gap_size
    
    # Chi components for real charges at the interface distance
    chi = _get_chi_components(fx, fy, f, ps, qs, z_int)
    
    pref = 2.0 * np.pi / (lx * ly)
    # Energy contribution: E_far = (dt * 2 * pi / Area) * sum(chi^2 / (2f))
    # Simplified: (dt * pi / Area) * sum(chi^2 / f)
    e_far = (dt * pref / 2.0) * np.sum(np.sum([c**2 for c in chi], axis=0) / f)
    
    return e_far

def get_elcic_energy(system, params: dict):
    box = np.array(system.box_l)
    lz_full = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    dt = params["delta_mid_top"]

    parts = system.part.all()
    qs, ps = parts.q.copy(), parts.pos.copy()

    # 1. Near-Field: Standard P3M for the primary charges
    # ELC logic dictates using the 3D solver on the primary box 
    # then correcting with the 2D sum.
    e_near = _get_config_energy(system, ps, qs, pref, eps, lz_full)

    # 2. Far-Field: Dielectric interaction
    e_far = _get_far_field_energy(box, gap, eps, qs, ps, dt) # TODO why is only delta_mid_bot passed? the top dielectric plane should also be relevant, right?

    return e_near + e_far

def _get_config_energy(system, p_set, q_set, prefactor, accuracy, lz):
    lx, ly = system.box_l[0], system.box_l[1]
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_set, q=q_set)
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]
    
    # Correction term logic remains consistent with ELC non-neutrality
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_set[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    e_corr = prefactor * fac * (xi1**2 - (lz**2 / 12.0) * xi0**2)
    
    system.electrostatics.clear()
    return e_3d + e_corr