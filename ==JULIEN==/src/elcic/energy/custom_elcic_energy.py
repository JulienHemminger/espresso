import espressomd
import espressomd.electrostatics
import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _elc_kernel(qs_L, xs_L, ys_L, zs_L,
                qs_R, xs_R, ys_R, zs_R,
                lx, ly, lz_box, gap_size, pw_error,
                prefactor):
    """
    Compute the ELC correction term  E_{2D+h} - E_{3D}  for a neutral (or
    non-neutral) system whose real-space box has height lz_box and whose
    gap is gap_size.

    The combined charge set is split into L< (lower half) and L> (upper
    half) as defined by the far formula.  Here we pass the two halves
    directly as (qs_L, ...) = L< and (qs_R, ...) = L>.

    Returns the scalar ELC correction energy (already multiplied by
    prefactor).

    NOTE: this is a *standalone* far-formula evaluation; it does NOT call
    P3M.  It implements Eq. (3.5) / (3.10) of Arnold et al. 2002.
    """
    ux = 1.0 / lx
    uy = 1.0 / ly
    uz = 1.0 / lz_box

    # ------------------------------------------------------------------
    # Build reciprocal-space grid
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Chi factors  χ^{±, s/c, s/c}_{L}(p,q)  =  Σ_i q_i exp(±2π f z_i)
    #              × sin/cos(ω_p x_i) × sin/cos(ω_q y_i)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # X-factors  X^{+, s/c, s/c}_{L<}(p,q)  and  X^{-, s/c, s/c}_{L>}
    # using Eq. (3.6) / (3.7):
    #   L'_{p,q}(z) = exp(-2π f z) / (1 - exp(-4 L_z f))
    # X^{+}_{L<} uses  L'(z_i)   (lower set, upper replica direction)
    # X^{-}_{L>} uses  L'(L_z - z_i) (upper set, lower replica direction)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ELC reciprocal correction  (Eq. 3.5, neutral case)
    # E_{2D+h} - E_{3D} |_{recip} =
    #   -½ ux uy Σ_{p,q} (1/f_{pq}) [χ^{-}_{L<} X^{+}_{L<}  cross-terms]
    #   -½ ux uy Σ_{p,q} (1/f_{pq}) [X^{-}_{L>} χ^{+}_{L>}  cross-terms]
    # ------------------------------------------------------------------
    def dot4(A, B, tags):
        return np.sum(inv_f * sum(A[t1] * B[t2] for t1, t2 in tags))

    cc_sc_cs_ss_RL = (('-cc', '+cc'), ('-sc', '+sc'), ('-cs', '+cs'), ('-ss', '+ss'))

    # Term 1: L< side  →  χ^{-}_{L<} · X^{+}_{L<}
    t1 = dot4(chiL, XL_plus,  cc_sc_cs_ss_RL)
    # Term 2: L> side  →  X^{-}_{L>} · χ^{+}_{L>}
    t2 = dot4(XR_minus, chiR, cc_sc_cs_ss_RL)

    e_recip = -0.5 * ux * uy * t1 - 0.5 * ux * uy * t2

    # ------------------------------------------------------------------
    # Dipole / non-neutral correction  (Eq. 3.8 / 3.9 / 3.10)
    # For a neutral system the xi0 terms vanish but we keep them general.
    # We operate on the FULL charge set (L ∪ R).
    # ------------------------------------------------------------------
    all_qs = np.concatenate([qs_L, qs_R])
    all_zs = np.concatenate([zs_L, zs_R])

    xi0 = np.sum(all_qs)
    xi1 = np.sum(all_qs * all_zs)
    xi2 = np.sum(all_qs * all_zs**2)

    # Eq. (3.8): replace standard dipole term with background-corrected one
    # Standard 3D dipole would be  2π u_x u_y u_z (xi1)^2
    # ELCIC/ELC replaces it with   2π u_x u_y u_z (xi1 - lz/2 · xi0)^2
    # The net change (what ELC *adds* relative to P3M's own dipole term) is:
    #   +2 ux uy uz (xi1)^2              ← subtract P3M's dipole contribution
    #   → actual correction keeps track only of difference
    # Following the ELC paper directly:
    e_dipole = 2.0 * ux * uy * uz * (xi1 - 0.5 * lz_box * xi0)**2
    # subtract the P3M dipole (which was  2π ux uy uz xi1^2  with tinfoil BC)
    e_dipole_correction = (
        2.0 * ux * uy * uz * ((xi1 - 0.5 * lz_box * xi0)**2 - xi1**2)
    )

    # Eq. (3.9): interaction of neutralizing background with charges
    e_bg = (
        -2.0 * ux * uy * uz * xi0 * xi2
        + 2.0 * ux * uy        * xi0 * xi1
        - (1.0/3.0) * ux * uy * lz_box * xi0**2
    )

    # Combined non-neutral / dipole term (Eq. 3.10 last two lines)
    e_nn = 2.0 * ux * uy * uz * (xi1**2 - xi0 * xi2 - (lz_box**2 / 12.0) * xi0**2)

    return prefactor * (e_recip + e_nn)


# ---------------------------------------------------------------------------
# ELC energy for a *single* system (P3M + ELC correction)
# ---------------------------------------------------------------------------

def _p3m_energy(system, qs, xs, ys, zs, lx, ly, lz_box,
                gap_size, pw_error, prefactor):
    """
    Run P3M on the particles currently in `system` and return the total
    electrostatic energy.  The caller is responsible for setting up
    particle positions / charges before calling this.
    """
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    e = system.analysis.energy()["total"]
    system.electrostatics.clear()
    return float(e)


# ---------------------------------------------------------------------------
# Far-formula interaction between L0 and L±2
# ---------------------------------------------------------------------------

def _compute_e_far(qs, xs, ys, zs, lx, ly, lz,
                   gap_size, pw_error, prefactor,
                   delta_b, delta_t):
    """
    Compute the interaction energy of the real charges L0 with the
    second-generation image charges L±2 using the far formula (Sec. IV.A).

    This implements Eqs. (4.6)–(4.12) of Tyagi et al. 2008, plugged into
    Eq. (3.4).

    Parameters
    ----------
    qs, xs, ys, zs : real charges and positions (all N particles)
    lx, ly, lz     : box dimensions of the *physical* slab (NOT the padded box)
    gap_size        : ELC gap
    pw_error        : plane-wave truncation error
    prefactor       : Coulomb prefactor
    delta_b, delta_t: dielectric contrast factors Δ_b, Δ_t
    """
    Delta = delta_b * delta_t   # Δ = Δ_b Δ_t

    # Subdivision threshold λ = gap_size
    lam = gap_size

    # Classify real charges into L0,−1 / L0,0 / L0,+1  (Eq. 4.3)
    mask_bot = zs <  lam               # L0,−1  : 0 < z < λ
    mask_top = zs >= lz - lam          # L0,+1  : lz−λ < z ≤ lz
    mask_mid = ~mask_bot & ~mask_top   # L0,0   : λ ≤ z ≤ lz−λ

    # ------------------------------------------------------------------
    # Build reciprocal-space grid
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helper: L_{p,q}(z) = Δ exp(-2π f z) / (1 - Δ exp(-4π l_z f))
    #         (Eq. 4.4 with generic Δ)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fourier (T) factors:  T_{p,q}(x_i, y_i) = [cc, sc, cs, ss]
    # shape of each: (N, n_modes)
    # ------------------------------------------------------------------
    cp = np.cos(omega_p * xs[:, None])
    sp = np.sin(omega_p * xs[:, None])
    cq = np.cos(omega_q * ys[:, None])
    sq = np.sin(omega_q * ys[:, None])

    T = {'cc': cp * cq, 'sc': sp * cq, 'cs': cp * sq, 'ss': sp * sq}

    # ------------------------------------------------------------------
    # χ factors for L0 (real charges) — Eq. (3.3)
    # χ^{±}_{L0}(p,q) = Σ_i q_i exp(±2π f z_i) T_{p,q}(x_i, y_i)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ---- L−2 factors  (Eqs. 4.6, 4.8, 4.9) -------------------------
    #
    # χ^{+}_{L−2}(p,q) =
    #   Σ_{i ∈ L0,−1} q_i [Δ_b L_{p,q}(2lz+z_i) + Δ L_{p,q}(2lz−z_i)] T_i
    # + Σ_{i ∈ L0,0∪L0,+1} q_i [Δ_b L_{p,q}(z_i) + Δ L_{p,q}(2lz−z_i)] T_i
    #
    # ξ^{(0)}_{L−2} =
    #   Σ_{i ∈ L0,−1} q_i/(1-Δ) (Δ_b Δ + Δ)
    # + Σ_{i ∈ L0,0∪L0,+1} q_i/(1-Δ) (Δ_b + Δ)
    #
    # ξ^{(1)}_{L−2} =
    #   Σ_{i ∈ L0,−1} q_i/(1-Δ) [−Δ_b Δ I(2lz+z_i) − Δ I(2lz−z_i)]
    # + Σ_{i ∈ L0,0∪L0,+1} q_i/(1-Δ) [−Δ_b I(z_i) − Δ I(2lz−z_i)]
    # ------------------------------------------------------------------

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
            # Note: Lpq_func already bakes in the Δ factor in the numerator.
            # But the series is:
            #   q_i Δ_b exp(-2π f z_i) + q_i Δ_b Δ exp(-2π f (2lz+z_i)) + ...
            #   = q_i Δ_b L_{p,q}(z_i)      with L containing Δ in numerator
            # For L0,−1 the *first* image (at -z_i) is already in L−1,
            # so the L−2 part starts from the 2nd term: position -(2lz+z_i)
            # → L(2lz+z_i) with numerator Δ_b Δ  (=Δ_b * Δ).
            # We handle this by re-building:
            #   Σ_m Δ^m exp(-2π f (2lz + z_i + 2m lz)) =
            #       Δ exp(-2π f (2lz+z_i)) / (1 - Δ exp(-4π f lz))
            # which is just Lpq_func(2lz+z_i, Delta) evaluated with Δ in numerator
            # (that's what Lpq_func computes).
            # The overall coefficient of this block is Δ_b (outer factor).
            # Similarly the "series 2.4" block:
            #   Σ_m Δ^{m+1} exp(-2π f (2lz + z_i + 2m lz)) starting m=0
            #   = Δ exp(-2π f (2lz - z_i)) / (1 - Δ exp(-4π f lz))   * outer=1? no
            # Let us re-examine from the paper carefully:
            # Series 2.3:  positions -(z_i), -(2lz+z_i), -(4lz+z_i), ...
            #              charges   q_i Δ_b, q_i Δ_b Δ, q_i Δ_b Δ^2, ...
            # L−1 contains only the first element -(z_i) for L0,−1.
            # L−2 gets the rest, starting from -(2lz+z_i):
            #   Σ_{m=1}^∞ q_i Δ_b Δ^{m-... wait, let's index properly:
            #   m=0: pos=−z_i, charge q_i Δ_b          → in L−1
            #   m=1: pos=−(2lz+z_i), charge q_i Δ_b Δ  → in L−2
            #   ...
            # So L−2 from series 2.3 (for L0,−1):
            #   Σ_{m=0}^∞ q_i Δ_b Δ^{m+1} exp(-2π f (2lz+z_i + 2m lz))
            #   = q_i Δ_b · Δ exp(-2π f (2lz+z_i)) / (1 - Δ exp(-4π f lz))
            #   = q_i Δ_b · Lpq_func(2lz+z_i, Delta)
            # Series 2.4: positions -(2lz-z_i), -(4lz-z_i), ...
            #             charges q_i Δ, q_i Δ^2, ...
            # All of series 2.4 is in L−2 (first element at -(2lz-z_i)):
            #   Σ_{m=0}^∞ q_i Δ^{m+1} exp(-2π f (2lz-z_i + 2m lz))
            #   = q_i · Lpq_func(2lz-z_i, Delta)
            coeff_bot = delta_b * Lm1 + Lm2   # shape (len(idx), n_modes)
            for tag, t in T.items():
                out[tag] += np.sum(qi[:, None] * coeff_bot * t[idx, :], axis=0)

        # --- L0,0 ∪ L0,+1 subset ---
        idx2 = np.where(mask_mid | mask_top)[0]
        if len(idx2) > 0:
            zi   = zs[idx2]
            qi   = qs[idx2]
            # For L0,0 and L0,+1: ALL of series 2.3 is in L−2
            #   (first image is at -z_i, but since z_i ≥ λ it is already > λ from bottom)
            # Series 2.3 (all terms):
            #   Σ_{m=0}^∞ q_i Δ_b Δ^m exp(-2π f (z_i + 2m lz))
            #   = q_i Δ_b Lpq_func(z_i, Delta)
            # Series 2.4 (all terms):
            #   Σ_{m=0}^∞ q_i Δ^{m+1} exp(-2π f (2lz-z_i + 2m lz))
            #   = q_i Lpq_func(2lz-z_i, Delta)
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

    # ------------------------------------------------------------------
    # ---- L+2 factors  (Eqs. 4.10, 4.11, 4.12) ----------------------
    #
    # χ^{−}_{L+2}(p,q) =
    #   Σ_{i ∈ L0,+1} q_i [Δ_t L_{p,q}(4lz−z_i) + Δ L_{p,q}(2lz+z_i)] T_i
    # + Σ_{i ∈ L0,0∪L0,−1} q_i [Δ_t L_{p,q}(2lz−z_i) + Δ L_{p,q}(2lz+z_i)] T_i
    #
    # ξ^{(0)}_{L+2} =
    #   Σ_{i ∈ L0,+1} q_i/(1-Δ) (Δ_t Δ + Δ)
    # + Σ_{i ∈ L0,0∪L0,−1} q_i/(1-Δ) (Δ_t + Δ)
    #
    # ξ^{(1)}_{L+2} =
    #   Σ_{i ∈ L0,+1} q_i/(1-Δ) [Δ_t Δ I(4lz−z_i) + Δ I(2lz+z_i)]
    # + Σ_{i ∈ L0,0∪L0,−1} q_i/(1-Δ) [Δ_t I(2lz−z_i) + Δ I(2lz+z_i)]
    # ------------------------------------------------------------------

    def chi_Lp2():
        """Returns χ^{−}_{L+2} as dict of (n_modes,) arrays."""
        out = {tag: np.zeros(len(f_pq)) for tag in ['cc', 'sc', 'cs', 'ss']}

        # --- L0,+1 subset ---
        idx = np.where(mask_top)[0]
        if len(idx) > 0:
            zi = zs[idx]
            qi = qs[idx]
            # Series 2.5: positions (2lz-z_i), (4lz-z_i), ...
            #             charges q_i Δ_t, q_i Δ_t Δ, ...
            # L+1 contains only the first (2lz-z_i) for L0,+1
            # L+2 from series 2.5 (L0,+1):
            #   Σ_{m=1}^∞ q_i Δ_t Δ^m exp(-2π f (4lz-z_i + 2m lz - 2lz))
            # Wait, let's be careful.  Positions in series 2.5:
            #   (2lz-z_i), (4lz-z_i), (6lz-z_i), ...
            # For the far formula we need exp(-2π f z') for z' > lz (above the box).
            # The L factor uses exp(-2π f (z' - lz)) to measure distance above lz.
            # Actually looking at Eq. (4.10):
            #   L+2 series 2.5 for L0,+1: starts at 2nd element = 4lz-z_i
            #   → Σ_{m=0}^∞ q_i Δ_t Δ^{m+1} exp(-2π f (4lz-z_i + 2m lz - lz))
            #   Hmm. Let's follow the paper's X^{-}_{L+2} definition directly.
            # From (4.10):
            #   χ^{-,s/c,s/c}_{L+2}(p,q) =
            #     Σ_{i∈L0,+1} q_i(Δ_t Δ L_{p,q}(4lz-z_i) + Δ L_{p,q}(2lz+z_i)) T
            #   + Σ_{i∈L0,0∪L0,-1} q_i(Δ_t L_{p,q}(2lz-z_i) + Δ L_{p,q}(2lz+z_i)) T
            # where L_{p,q}(z) = Δ exp(-2π f z) / (1 - Δ exp(-4π f lz))
            # Note: the coefficients Δ_t Δ and Δ_t are the outer multiplicative factors
            # (they do NOT go inside the Lpq_func which already has its own Δ).
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

    # ------------------------------------------------------------------
    # Assemble via Eq. (3.4):
    #   E(L0, L±2) = ½ ux uy Σ_{p,q≠0} (1/f_{p,q}) [
    #       χ^{-}_{L0} χ^{+}_{L±2}  cross-terms ]
    #              + (dipole cross terms between L0 and L±2)
    #
    # More precisely Eq. (3.4):
    #   E = ½ ux uy Σ (1/f) [χ^{-cc}_{L>} χ^{+cc}_{L<} + ... ]
    #             - π ux uy ξ^{(1)}_{L>} ξ^{(0)}_{L<}
    #             - π ux uy ξ^{(0)}_{L>} ξ^{(1)}_{L<}
    # where L< is L0 (lower half, smaller z) and L> is L±2 (upper half).
    # But here the ">" / "<" ordering depends on which image group.
    #
    # For L−2 (images BELOW the box):  L> = L0, L< = L−2
    #   E(L0, L−2) = ½ ux uy Σ (1/f) [χ^{-}_{L0} χ^{+}_{L−2} cross-terms]
    #              - π ux uy (ξ1_{L0} ξ0_{L−2} + ξ0_{L0} ξ1_{L−2})
    #
    # For L+2 (images ABOVE the box):  L> = L+2, L< = L0
    #   E(L0, L+2) = ½ ux uy Σ (1/f) [χ^{-}_{L+2} χ^{+}_{L0} cross-terms]
    #              - π ux uy (ξ1_{L+2} ξ0_{L0} + ξ0_{L+2} ξ1_{L0})
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper: run P3M + standalone ELC correction on an arbitrary charge set
#         that has been loaded into `system`.
# ---------------------------------------------------------------------------

def _run_elc_on_system(system, lx, ly, lz_phys, lz_padded, gap_size,
                       pw_error, prefactor, description=""):
    """
    Returns E_{2D+h} for the particles currently in `system`, computed as:
        E_{2D+h} = E_{P3M}(lz_padded) + ELC_correction(lz_padded, gap_size)

    `lz_phys`   – physical slab height (height of the charge region)
    `lz_padded` – box height used for P3M (= lz_phys + 3*gap_size)
    """
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    e_3d = float(system.analysis.energy()["total"])
    system.electrostatics.clear()

    parts = system.part.all()
    qs_s = parts.q
    pos_s = parts.pos
    xs_s, ys_s, zs_s = pos_s[:, 0], pos_s[:, 1], pos_s[:, 2]

    # ELC correction: we need to split the charge set into lower and upper
    # halves for the far formula; but here we use _elc_kernel which
    # handles it internally using the full set (neutral case).
    # For simplicity we pass the full set as both halves and let the kernel
    # handle the correct split internally.
    # Actually _elc_kernel expects pre-split L< and L>, so let's split at lz/2.
    mid = lz_padded / 2.0
    mask_lo = zs_s <= mid
    mask_hi = ~mask_lo

    e_elc = _elc_kernel(
        qs_s[mask_lo], xs_s[mask_lo], ys_s[mask_lo], zs_s[mask_lo],
        qs_s[mask_hi], xs_s[mask_hi], ys_s[mask_hi], zs_s[mask_hi],
        lx, ly, lz_padded, gap_size, pw_error, prefactor
    )

    return e_3d + e_elc, e_3d, e_elc


# ---------------------------------------------------------------------------
# Main ELCIC functions
# ---------------------------------------------------------------------------

def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    """
    Compute ELCIC energy contributions for a 2D+h slab system with two
    dielectric interfaces.

    The method follows Tyagi, Arnold, Holm, J. Chem. Phys. 129, 204102 (2008).

    Parameters
    ----------
    system        : ESPResSo system with particles already set up.
    gap_size      : ELC gap size λ (must be > 0; charges must stay outside
                    [0,λ) and (lz-λ, lz]).
    pw_error      : plane-wave truncation error target.
    prefactor     : Coulomb prefactor (= 1/(4πε₀ εm)).
    delta_mid_bot : Δ_b = (ε_m − ε_b)/(ε_m + ε_b)
    delta_mid_top : Δ_t = (ε_m − ε_t)/(ε_m + ε_t)

    Returns
    -------
    dict with keys:
        "e_near"   – E(L0, L0 ∪ L±1) computed via P3M + ELC on LT
        "e_far"    – E(L0, L±2) computed via far formula
        "l0"       – sub-dict with contributions from the L0-only ELC term
        "pm1"      – sub-dict with contributions from the L±1 ELC term
        "lt"       – sub-dict with contributions from the LT ELC term
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q.copy()
    pos = parts.pos.copy()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    Delta     = delta_mid_bot * delta_mid_top

    # ------------------------------------------------------------------
    # 1.  E_far: interaction of L0 with L±2  (far formula, Sec. IV.A)
    # ------------------------------------------------------------------
    e_far_total, e_far_detail = _compute_e_far(
        qs, xs, ys, zs,
        lx, ly, lz,
        gap_size, pw_error, prefactor,
        delta_mid_bot, delta_mid_top,
    )

    # ------------------------------------------------------------------
    # 2.  Build LT = L−1 ∪ L0 ∪ L+1 and compute E(LT, LT), E(L±1, L±1),
    #     E(L0, L0) using P3M + ELC, then obtain E(L0, LT) via Eq. (4.14).
    #
    # The padded box has height lz_padded = lz + 3*gap_size so that there
    # is a gap of at least gap_size between the real charges (and their
    # near images) and their periodic replicas.
    # ------------------------------------------------------------------
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

    # ---- L−1: first image of L0,−1 through lower interface (series 2.3 m=0)
    #     position: -z_i → inside lower dielectric, but we shift z periodically
    #     For the P3M box of height lz_padded, images below z=0 are placed at
    #     negative z, which we fold back: z_image = -z_i → shifted to lz_padded - z_i
    #     Actually we keep the physical meaning: the image is at z' = -z_i.
    #     Since lz_padded includes a gap, we need z' > 0 in the padded box.
    #     The padded box starts at z=0 (real charges) and the gap is at the TOP.
    #     We shift the entire system so real charges live in [gap_size, lz+gap_size]
    #     inside the padded box, making room for images below and above.
    #     → shift: z_shifted = z + gap_size
    #
    # RE-READING THE PAPER (Sec. IV.B):
    #   LT is placed in a box of size lx × ly × (lz + 3λ) such that a gap λ
    #   remains.  The real charges L0 are shifted to [λ, lz+λ], L−1 images go
    #   to [0, λ], and L+1 images go to [lz+λ, lz+2λ].  The gap [lz+2λ, lz+3λ]
    #   is empty (the ELC gap).
    # ------------------------------------------------------------------

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

    # L+1 images (above upper interface at z=lz, i.e. at 2lz-z_i physically)
    # In shifted coords: 2lz - z_i + z_shift = 2lz - z_i + gap_size
    # For z_i in [lz-gap_size, lz]: 2lz-z_i in [lz, lz+gap_size]
    # shifted: 2lz-z_i+gap_size in [lz+gap_size, lz+2*gap_size] ✓
    if len(idx_top) > 0:
        zs_Lp1_shifted = 2.0 * lz - zs[idx_top] + z_shift
    else:
        zs_Lp1_shifted = np.array([])

    # ------------------------------------------------------------------
    # Build three particle sets for ESPResSo:
    #   (a) LT = L0 ∪ L−1 ∪ L+1  (all in shifted coords, padded box)
    #   (b) L±1 only
    #   (c) L0 only  (shifted)
    # For each we spin up P3M + ELC and get the 2D+h energy.
    # ------------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # (a) E(LT, LT)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # (b) E(L±1, L±1)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # (c) E(L0, L0)  — real charges only, shifted into padded box
    # ----------------------------------------------------------------
    _setup_particles(system, lz_padded, qs, xs, ys, zs_shifted)
    e_L0_total, e_L0_3d, e_L0_elc = _run_elc_on_system(
        system, lx, ly, lz, lz_padded, gap_size, pw_error, prefactor, "L0"
    )

    # ----------------------------------------------------------------
    # Restore original system
    # ----------------------------------------------------------------
    system.part.clear()
    system.box_l = orig_box
    for i, (q_i, p_i) in enumerate(zip(orig_qs, orig_pos)):
        system.part.add(id=i, pos=p_i.tolist(), q=float(q_i))

    # ----------------------------------------------------------------
    # E(L0, LT) via Eq. (4.14):
    #   Φ(L0, LT) = ½ [Φ(LT,LT) − Φ(L±1, L±1) + Φ(L0, L0)]
    # ----------------------------------------------------------------
    e_near = 0.5 * (e_LT_total - e_L1_total + e_L0_total)

    # ------------------------------------------------------------------
    # Assemble output dict
    # ------------------------------------------------------------------
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
    """
    Compute the total ELCIC electrostatic energy of the system.

    Returns
    -------
    float – total energy E = E_near + E_far
    """
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]