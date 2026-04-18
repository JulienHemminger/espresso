import espressomd
import espressomd.electrostatics
import numpy as np


def _elc_kernel(qs_L, xs_L, ys_L, zs_L,
                qs_R, xs_R, ys_R, zs_R,
                lx, ly, lz_box, gap_size, pw_error,
                prefactor):
  
    ux = 1.0 / lx
    uy = 1.0 / ly
    uz = 1.0 / lz_box

    
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_arr = np.arange(-p_max, p_max + 1)
    q_arr = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_arr, q_arr)
    P, Q = P.flatten(), Q.flatten()

    mask = ((P != 0) | (Q != 0)) & (
        np.sqrt((P * ux) ** 2 + (Q * uy) ** 2) <= f_max
    )
    P, Q = P[mask], Q[mask]
    fx = P * ux          # p/lx
    fy = Q * uy          # q/ly
    f  = np.sqrt(fx**2 + fy**2)   # f_{pq}

    omega_p = 2.0 * np.pi * fx    # 2π p/lx
    omega_q = 2.0 * np.pi * fy    # 2π q/ly
    two_pi_f = 2.0 * np.pi * f    # 2π f_{pq}

   
    def chi_factors(qs, xs, ys, zs):
        """Returns dict with keys (+cc, +sc, +cs, +ss, -cc, -sc, -cs, -ss)
        each an array of shape (n_modes,)."""
        ex_p = np.exp( two_pi_f * zs[:, None])   # (N, n_modes)
        ex_m = np.exp(-two_pi_f * zs[:, None])
        c_p  = np.cos(omega_p * xs[:, None])
        s_p  = np.sin(omega_p * xs[:, None])
        c_q  = np.cos(omega_q * ys[:, None])
        s_q  = np.sin(omega_q * ys[:, None])
        w    = qs[:, None]

        out = {}
        for sign, ex, tag in [('+', ex_p, '+'), ('-', ex_m, '-')]:
            out[tag + 'cc'] = np.sum(w * ex * c_p * c_q, axis=0)
            out[tag + 'sc'] = np.sum(w * ex * s_p * c_q, axis=0)
            out[tag + 'cs'] = np.sum(w * ex * c_p * s_q, axis=0)
            out[tag + 'ss'] = np.sum(w * ex * s_p * s_q, axis=0)
        return out

    chiL = chi_factors(qs_L, xs_L, ys_L, zs_L)
    chiR = chi_factors(qs_R, xs_R, ys_R, zs_R)

   
    denom = 1.0 - np.exp(-4.0 * np.pi * f * lz_box)   # (n_modes,)

    def Lpq(z_scalar):
        """Scalar version of L'_{p,q}(z) for a single z."""
        return np.exp(-two_pi_f * z_scalar) / denom

    def X_plus_factors(qs, xs, ys, zs):
        """X^{+, s/c, s/c}_{L<} — uses L'(z_i)."""
        Lp = np.exp(-two_pi_f * zs[:, None]) / denom[None, :]   # (N, modes)
        c_p = np.cos(omega_p * xs[:, None])
        s_p = np.sin(omega_p * xs[:, None])
        c_q = np.cos(omega_q * ys[:, None])
        s_q = np.sin(omega_q * ys[:, None])
        w   = qs[:, None]
        out = {}
        out['+cc'] = np.sum(w * Lp * c_p * c_q, axis=0)
        out['+sc'] = np.sum(w * Lp * s_p * c_q, axis=0)
        out['+cs'] = np.sum(w * Lp * c_p * s_q, axis=0)
        out['+ss'] = np.sum(w * Lp * s_p * s_q, axis=0)
        return out

    def X_minus_factors(qs, xs, ys, zs):
        """X^{-, s/c, s/c}_{L>} — uses L'(L_z - z_i)."""
        Lp = np.exp(-two_pi_f * (lz_box - zs[:, None])) / denom[None, :]
        c_p = np.cos(omega_p * xs[:, None])
        s_p = np.sin(omega_p * xs[:, None])
        c_q = np.cos(omega_q * ys[:, None])
        s_q = np.sin(omega_q * ys[:, None])
        w   = qs[:, None]
        out = {}
        out['-cc'] = np.sum(w * Lp * c_p * c_q, axis=0)
        out['-sc'] = np.sum(w * Lp * s_p * c_q, axis=0)
        out['-cs'] = np.sum(w * Lp * c_p * s_q, axis=0)
        out['-ss'] = np.sum(w * Lp * s_p * s_q, axis=0)
        return out

    XL_plus  = X_plus_factors (qs_L, xs_L, ys_L, zs_L)
    XR_minus = X_minus_factors(qs_R, xs_R, ys_R, zs_R)

    inv_f = 1.0 / f    # (n_modes,)

   
    def dot4(A, B, tags):
        return np.sum(inv_f * sum(A[t1] * B[t2] for t1, t2 in tags))

    cc_sc_cs_ss_RL = (('-cc', '+cc'), ('-sc', '+sc'), ('-cs', '+cs'), ('-ss', '+ss'))

    # Term 1: L< side  →  χ^{-}_{L<} · X^{+}_{L<}
    t1 = dot4(chiL, XL_plus,  cc_sc_cs_ss_RL)
    # Term 2: L> side  →  X^{-}_{L>} · χ^{+}_{L>}
    t2 = dot4(XR_minus, chiR, cc_sc_cs_ss_RL)

    e_recip = -0.5 * ux * uy * t1 - 0.5 * ux * uy * t2

   
    all_qs = np.concatenate([qs_L, qs_R])
    all_zs = np.concatenate([zs_L, zs_R])

    xi0 = np.sum(all_qs)
    xi1 = np.sum(all_qs * all_zs)
    xi2 = np.sum(all_qs * all_zs**2)

    
    e_dipole = 2.0 * ux * uy * uz * (xi1 - 0.5 * lz_box * xi0)**2
    e_dipole_correction = (
        2.0 * ux * uy * uz * ((xi1 - 0.5 * lz_box * xi0)**2 - xi1**2)
    )

    e_bg = (
        -2.0 * ux * uy * uz * xi0 * xi2
        + 2.0 * ux * uy        * xi0 * xi1
        - (1.0/3.0) * ux * uy * lz_box * xi0**2
    )

    # Combined non-neutral / dipole term (Eq. 3.10 last two lines)
    e_nn = 2.0 * ux * uy * uz * (xi1**2 - xi0 * xi2 - (lz_box**2 / 12.0) * xi0**2)

    return prefactor * (e_recip + e_nn)

def _p3m_energy(system, qs, xs, ys, zs, lx, ly, lz_box,
                gap_size, pw_error, prefactor):
   
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e = system.analysis.energy()["total"]
    return float(e)

def _compute_e_far(qs, xs, ys, zs, lx, ly, lz,
                   gap_size, pw_error, prefactor,
                   delta_b, delta_t):
 
    Delta = delta_b * delta_t   # Δ = Δ_b Δ_t

    # Subdivision threshold λ = gap_size
    lam = gap_size

    # Classify real charges into L0,−1 / L0,0 / L0,+1  (Eq. 4.3)
    mask_bot = zs <  lam               # L0,−1  : 0 < z < λ
    mask_top = zs >= lz - lam          # L0,+1  : lz−λ < z ≤ lz
    mask_mid = ~mask_bot & ~mask_top   # L0,0   : λ ≤ z ≤ lz−λ

    f_max = -np.log(pw_error) / (2.0 * np.pi * lam)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p_arr = np.arange(-p_max, p_max + 1)
    q_arr = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p_arr, q_arr)
    P, Q = P.flatten(), Q.flatten()

    ux = 1.0 / lx
    uy = 1.0 / ly

    mask_k = ((P != 0) | (Q != 0)) & (
        np.sqrt((P * ux) ** 2 + (Q * uy) ** 2) <= f_max
    )
    P, Q  = P[mask_k], Q[mask_k]
    fx    = P * ux
    fy    = Q * uy
    f_pq  = np.sqrt(fx**2 + fy**2)

    omega_p   = 2.0 * np.pi * fx
    omega_q   = 2.0 * np.pi * fy
    two_pi_f  = 2.0 * np.pi * f_pq

    
    def Lpq_func(z_arr, Delta_val):
        """
        Vectorised: z_arr shape (N,) → output shape (N, n_modes).
        """
        num   = Delta_val * np.exp(-two_pi_f * z_arr[:, None])
        denom = 1.0 - Delta_val * np.exp(-4.0 * np.pi * f_pq * lz)
        return num / denom[None, :]

    # I(z) = 1/(1-Δ) * (z + 2 l_z Δ/(1-Δ))  (Eq. 4.5)
    def Iz_func(z_arr, Delta_val):
        """shape (N,)"""
        return (1.0 / (1.0 - Delta_val)) * (z_arr + 2.0 * lz * Delta_val / (1.0 - Delta_val))

   
    cp = np.cos(omega_p * xs[:, None])
    sp = np.sin(omega_p * xs[:, None])
    cq = np.cos(omega_q * ys[:, None])
    sq = np.sin(omega_q * ys[:, None])

    T = {'cc': cp * cq, 'sc': sp * cq, 'cs': cp * sq, 'ss': sp * sq}

    
    def chi_pm(sign):
        ex = np.exp(sign * two_pi_f * zs[:, None])
        out = {}
        for tag, t in T.items():
            out[tag] = np.sum(qs[:, None] * ex * t, axis=0)
        return out

    chi_p_L0 = chi_pm(+1.0)   # χ^{+}_{L0}
    chi_m_L0 = chi_pm(-1.0)   # χ^{-}_{L0}

    # scalar moments of L0
    xi0_L0 = float(np.sum(qs))
    xi1_L0 = float(np.sum(qs * zs))

   

    def chi_Lm2():
        """Returns χ^{+}_{L−2} as dict of (n_modes,) arrays."""
        out = {tag: np.zeros(len(f_pq)) for tag in ['cc', 'sc', 'cs', 'ss']}

        # --- L0,−1 subset ---
        idx = np.where(mask_bot)[0]
        if len(idx) > 0:
            zi   = zs[idx]
            qi   = qs[idx]
            Lm1  = Lpq_func(2.0 * lz + zi, Delta)   # L_{p,q}(2lz+z_i)  [Δ already inside]
            Lm2  = Lpq_func(2.0 * lz - zi, Delta)   # L_{p,q}(2lz−z_i)
           
            coeff_bot = delta_b * Lm1 + Lm2   # shape (len(idx), n_modes)
            for tag, t in T.items():
                out[tag] += np.sum(qi[:, None] * coeff_bot * t[idx, :], axis=0)

        # --- L0,0 ∪ L0,+1 subset ---
        idx2 = np.where(mask_mid | mask_top)[0]
        if len(idx2) > 0:
            zi   = zs[idx2]
            qi   = qs[idx2]
           
            Lm1  = Lpq_func(zi,           Delta)
            Lm2  = Lpq_func(2.0 * lz - zi, Delta)
            coeff_mid = delta_b * Lm1 + Lm2
            for tag, t in T.items():
                out[tag] += np.sum(qi[:, None] * coeff_mid * t[idx2, :], axis=0)

        return out

    # ξ^{(0)}_{L−2}  (Eq. 4.8)
    def xi0_Lm2():
        s = 0.0
        idx_bot = np.where(mask_bot)[0]
        if len(idx_bot) > 0:
            s += np.sum(qs[idx_bot]) * (delta_b * Delta + Delta) / (1.0 - Delta)
        idx_rest = np.where(mask_mid | mask_top)[0]
        if len(idx_rest) > 0:
            s += np.sum(qs[idx_rest]) * (delta_b + Delta) / (1.0 - Delta)
        return float(s)

    # ξ^{(1)}_{L−2}  (Eq. 4.9)
    def xi1_Lm2():
        s = 0.0
        idx_bot = np.where(mask_bot)[0]
        if len(idx_bot) > 0:
            zi = zs[idx_bot]
            qi = qs[idx_bot]
            # I(2lz+z_i) and I(2lz-z_i)
            Ip  = Iz_func(2.0 * lz + zi, Delta)
            Im  = Iz_func(2.0 * lz - zi, Delta)
            s  += np.sum(qi * (-delta_b * Delta * Ip - Delta * Im) / (1.0 - Delta))
        idx_rest = np.where(mask_mid | mask_top)[0]
        if len(idx_rest) > 0:
            zi = zs[idx_rest]
            qi = qs[idx_rest]
            Ip  = Iz_func(zi,             Delta)
            Im  = Iz_func(2.0 * lz - zi,  Delta)
            s  += np.sum(qi * (-delta_b * Ip - Delta * Im) / (1.0 - Delta))
        return float(s)

   

    def chi_Lp2():
        """Returns χ^{−}_{L+2} as dict of (n_modes,) arrays."""
        out = {tag: np.zeros(len(f_pq)) for tag in ['cc', 'sc', 'cs', 'ss']}

        # --- L0,+1 subset ---
        idx = np.where(mask_top)[0]
        if len(idx) > 0:
            zi = zs[idx]
            qi = qs[idx]
            
            Lp1  = Lpq_func(4.0 * lz - zi, Delta)
            Lp2  = Lpq_func(2.0 * lz + zi, Delta)
            coeff = delta_t * Delta * Lp1 + Delta * Lp2
            for tag, t in T.items():
                out[tag] += np.sum(qi[:, None] * coeff * t[idx, :], axis=0)

        # --- L0,0 ∪ L0,−1 subset ---
        idx2 = np.where(mask_mid | mask_bot)[0]
        if len(idx2) > 0:
            zi = zs[idx2]
            qi = qs[idx2]
            Lp1  = Lpq_func(2.0 * lz - zi, Delta)
            Lp2  = Lpq_func(2.0 * lz + zi, Delta)
            coeff = delta_t * Lp1 + Delta * Lp2
            for tag, t in T.items():
                out[tag] += np.sum(qi[:, None] * coeff * t[idx2, :], axis=0)

        return out

    # ξ^{(0)}_{L+2}  (Eq. 4.11)
    def xi0_Lp2():
        s = 0.0
        idx_top = np.where(mask_top)[0]
        if len(idx_top) > 0:
            s += np.sum(qs[idx_top]) * (delta_t * Delta + Delta) / (1.0 - Delta)
        idx_rest = np.where(mask_mid | mask_bot)[0]
        if len(idx_rest) > 0:
            s += np.sum(qs[idx_rest]) * (delta_t + Delta) / (1.0 - Delta)
        return float(s)

    # ξ^{(1)}_{L+2}  (Eq. 4.12)
    def xi1_Lp2():
        s = 0.0
        idx_top = np.where(mask_top)[0]
        if len(idx_top) > 0:
            zi = zs[idx_top]
            qi = qs[idx_top]
            Ip = Iz_func(4.0 * lz - zi, Delta)
            Im = Iz_func(2.0 * lz + zi, Delta)
            s += np.sum(qi * (delta_t * Delta * Ip + Delta * Im) / (1.0 - Delta))
        idx_rest = np.where(mask_mid | mask_bot)[0]
        if len(idx_rest) > 0:
            zi = zs[idx_rest]
            qi = qs[idx_rest]
            Ip = Iz_func(2.0 * lz - zi, Delta)
            Im = Iz_func(2.0 * lz + zi, Delta)
            s += np.sum(qi * (delta_t * Ip + Delta * Im) / (1.0 - Delta))
        return float(s)

    inv_f = 1.0 / f_pq

    chi_p_Lm2 = chi_Lm2()   # χ^{+}_{L−2}
    chi_m_Lp2 = chi_Lp2()   # χ^{−}_{L+2}

    xi0_m2 = xi0_Lm2()
    xi1_m2 = xi1_Lm2()
    xi0_p2 = xi0_Lp2()
    xi1_p2 = xi1_Lp2()

    def cross4(A, B):
        """½ ux uy Σ (1/f) [cc+sc+cs+ss cross-terms]."""
        s = np.sum(inv_f * (
            A['cc'] * B['cc'] +
            A['sc'] * B['sc'] +
            A['cs'] * B['cs'] +
            A['ss'] * B['ss']
        ))
        return 0.5 * ux * uy * s

    # E(L0, L−2): χ^{-}_{L0} (higher z) × χ^{+}_{L−2} (lower z)
    e_recip_m2 = cross4(chi_m_L0, chi_p_Lm2)
    e_dipole_m2 = -np.pi * ux * uy * (
        xi1_L0 * xi0_m2 + xi0_L0 * xi1_m2
    )

    # E(L0, L+2): χ^{-}_{L+2} (higher z) × χ^{+}_{L0} (lower z)
    e_recip_p2 = cross4(chi_m_Lp2, chi_p_L0)
    e_dipole_p2 = -np.pi * ux * uy * (
        xi1_p2 * xi0_L0 + xi0_p2 * xi1_L0
    )

    e_far_total = e_recip_m2 + e_dipole_m2 + e_recip_p2 + e_dipole_p2

    return prefactor * e_far_total, {
        "e_recip_m2":   prefactor * e_recip_m2,
        "e_dipole_m2":  prefactor * e_dipole_m2,
        "e_recip_p2":   prefactor * e_recip_p2,
        "e_dipole_p2":  prefactor * e_dipole_p2,
    }

def _run_elc_on_system(system, lx, ly, lz_phys, lz_padded, gap_size,
                       pw_error, prefactor, description=""):
   
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = float(system.analysis.energy()["total"])

    parts = system.part.all()
    qs_s = parts.q
    pos_s = parts.pos
    xs_s, ys_s, zs_s = pos_s[:, 0], pos_s[:, 1], pos_s[:, 2]

    mid = lz_padded / 2.0
    mask_lo = zs_s <= mid
    mask_hi = ~mask_lo

    e_elc = _elc_kernel(
        qs_s[mask_lo], xs_s[mask_lo], ys_s[mask_lo], zs_s[mask_lo],
        qs_s[mask_hi], xs_s[mask_hi], ys_s[mask_hi], zs_s[mask_hi],
        lx, ly, lz_padded, gap_size, pw_error, prefactor
    )

    return e_3d + e_elc, e_3d, e_elc

def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
   
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q.copy()
    pos = parts.pos.copy()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    Delta     = delta_mid_bot * delta_mid_top

   
    e_far_total, e_far_detail = _compute_e_far(
        qs, xs, ys, zs,
        lx, ly, lz,
        gap_size, pw_error, prefactor,
        delta_mid_bot, delta_mid_top,
    )

    
    lz_padded = lz + 3.0 * gap_size

    # Classify particles
    mask_bot_real = zs < gap_size                     # L0,−1
    mask_top_real = zs >= lz - gap_size               # L0,+1
    mask_mid_real = ~mask_bot_real & ~mask_top_real   # L0,0

    # ---- L+1: first image of L0,+1 through upper interface (series 2.5 m=0)
    #     position: 2lz - z_i,  charge: q_i Δ_t
    idx_top = np.where(mask_top_real)[0]
    if len(idx_top) > 0:
        qs_Lp1 = delta_mid_top * qs[idx_top]
        xs_Lp1 = xs[idx_top]
        ys_Lp1 = ys[idx_top]
        zs_Lp1 = 2.0 * lz - zs[idx_top]   # above the box
    else:
        qs_Lp1 = np.array([])
        xs_Lp1 = np.array([])
        ys_Lp1 = np.array([])
        zs_Lp1 = np.array([])

   

    z_shift = gap_size  # shift real charges up by one gap_size

    # Shifted real charges
    zs_shifted = zs + z_shift   # now in [gap_size, lz+gap_size]

    # L−1 images (below lower interface at z=0, i.e. at z_shifted < 0 → folded)
    idx_bot = np.where(mask_bot_real)[0]
    if len(idx_bot) > 0:
        qs_Lm1 = delta_mid_bot * qs[idx_bot]
        xs_Lm1 = xs[idx_bot]
        ys_Lm1 = ys[idx_bot]
        # image at -z_i in physical coords → in shifted coords: -z_i + z_shift
        # but -z_i + gap_size = gap_size - z_i; since z_i < gap_size this is >0
        zs_Lm1 = z_shift - zs[idx_bot]   # in [0, gap_size)
    else:
        qs_Lm1 = np.array([])
        xs_Lm1 = np.array([])
        ys_Lm1 = np.array([])
        zs_Lm1 = np.array([])

   
    if len(idx_top) > 0:
        zs_Lp1_shifted = 2.0 * lz - zs[idx_top] + z_shift
    else:
        zs_Lp1_shifted = np.array([])

   

    # Save original box and particles
    orig_box = system.box_l.copy()
    orig_qs  = qs.copy()
    orig_pos = pos.copy()

    def _setup_particles(system, lz_new,
                         q_list, x_list, y_list, z_list):
        """Replace all particles with the given set in a box lx × ly × lz_new."""
        system.part.clear()
        system.box_l = [lx, ly, lz_new]
        if len(q_list) > 0:
            for i, (q_i, x_i, y_i, z_i) in enumerate(
                zip(q_list, x_list, y_list, z_list)
            ):
                system.part.add(id=i, pos=[x_i, y_i, z_i], q=float(q_i))

   
    qs_LT = np.concatenate([qs_Lm1, qs,      qs_Lp1])
    xs_LT = np.concatenate([xs_Lm1, xs,      xs_Lp1 if len(xs_Lp1) > 0 else np.array([])])
    ys_LT = np.concatenate([ys_Lm1, ys,      ys_Lp1 if len(ys_Lp1) > 0 else np.array([])])
    zs_LT = np.concatenate([zs_Lm1, zs_shifted,
                             zs_Lp1_shifted if len(zs_Lp1_shifted) > 0 else np.array([])])

    _setup_particles(system, lz_padded, qs_LT, xs_LT, ys_LT, zs_LT)

    # ELC gap for LT is gap_size (the empty region at the top of the padded box)
    e_LT_total, e_LT_3d, e_LT_elc = _run_elc_on_system(
        system, lx, ly, lz, lz_padded, gap_size, pw_error, prefactor, "LT"
    )

    
    qs_L1 = np.concatenate([qs_Lm1, qs_Lp1])
    xs_L1 = np.concatenate([xs_Lm1, xs_Lp1 if len(xs_Lp1) > 0 else np.array([])])
    ys_L1 = np.concatenate([ys_Lm1, ys_Lp1 if len(ys_Lp1) > 0 else np.array([])])
    zs_L1 = np.concatenate([zs_Lm1,
                             zs_Lp1_shifted if len(zs_Lp1_shifted) > 0 else np.array([])])

    if len(qs_L1) > 0:
        _setup_particles(system, lz_padded, qs_L1, xs_L1, ys_L1, zs_L1)
        e_L1_total, e_L1_3d, e_L1_elc = _run_elc_on_system(
            system, lx, ly, lz, lz_padded, gap_size, pw_error, prefactor, "L1"
        )
    else:
        e_L1_total = 0.0
        e_L1_3d   = 0.0
        e_L1_elc  = 0.0

   
    _setup_particles(system, lz_padded, qs, xs, ys, zs_shifted)
    e_L0_total, e_L0_3d, e_L0_elc = _run_elc_on_system(
        system, lx, ly, lz, lz_padded, gap_size, pw_error, prefactor, "L0"
    )

    
    system.part.clear()
    system.box_l = orig_box
    for i, (q_i, p_i) in enumerate(zip(orig_qs, orig_pos)):
        system.part.add(id=i, pos=p_i.tolist(), q=float(q_i))

   
    e_near = 0.5 * (e_LT_total - e_L1_total + e_L0_total)

   
    contribs = {
        "e_near": float(e_near),
        "e_far":  float(e_far_total),
        # Sub-breakdowns for diagnostics
        "l0": {
            "e_3d":   e_L0_3d,
            "e_elc":  e_L0_elc,
            "total":  e_L0_total,
        },
        "pm1": {
            "e_3d":   e_L1_3d,
            "e_elc":  e_L1_elc,
            "total":  e_L1_total,
        },
        "lt": {
            "e_3d":   e_LT_3d,
            "e_elc":  e_LT_elc,
            "total":  e_LT_total,
        },
        "e_far_detail": e_far_detail,
    }
    return contribs


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]