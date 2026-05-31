import espressomd
import espressomd.electrostatics
import numpy as np

def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP/BOTTOM dielectric interfaces using ELCIC.
    """
    box = np.array(system.box_l)
    lx, ly, lz_full = box[0], box[1], box[2]
    gap = params["gap_size"]
    pw_error = params["pw_error"]
    pref = params.get("prefactor", 1.0)
    
    db = params["delta_mid_bot"]
    dt = params["delta_mid_top"]
    
    # Avoid exact 1.0 to prevent division by zero in analytical sum limits.
    delta = np.clip(db * dt, -0.99999, 0.99999) 
    
    lz = lz_full - gap
    lambda_ = params.get("lambda", lz / 2)
    lambda_ = np.clip(lambda_, 1e-3, lz / 2)
    
    q_arr = params["charges"]
    pos_arr = params["positions"]

    # --- Part 1: Partitioning the charges (Section IV) ---
    L0_neg1, L0_0, L0_pos1 = [], [], []
    L0 = []
    
    for q_i, p_i in zip(q_arr, pos_arr):
        p_i_arr = np.array(p_i)
        L0.append((q_i, p_i_arr))
        if p_i_arr[2] <= lambda_:
            L0_neg1.append((q_i, p_i_arr))
        elif p_i_arr[2] <= lz - lambda_:
            L0_0.append((q_i, p_i_arr))
        else:
            L0_pos1.append((q_i, p_i_arr))

    # First generation image charges
    L_neg1 = []
    for q_i, p_i in L0_neg1:
        L_neg1.append((q_i * db, np.array([p_i[0], p_i[1], -p_i[2]])))

    L_pos1 = []
    for q_i, p_i in L0_pos1:
        L_pos1.append((q_i * dt, np.array([p_i[0], p_i[1], 2 * lz - p_i[2]])))

    L_T = L_neg1 + L0 + L_pos1

    # --- Part 2: Evaluate Near Interactions (L_T, L_0, L_pm1) via standard ELC ---
    # Shift coordinate origin by +lambda_ so z-coordinates reside strictly within [0, lz + 2*lambda_]
    shift_z = lambda_
    box_z_T = lz + 3 * lambda_
    gap_T = lambda_ # Required gap size matching artificial box dimensions

    def shifted(lst):
        return [(qi, np.array([pi[0], pi[1], pi[2] + shift_z])) for qi, pi in lst]

    def calc_elc_energy(charge_pos, box_z_val, gap_val):
        if not charge_pos:
            return 0.0
        system.part.clear()
        system.box_l = [lx, ly, box_z_val]
        for qi, pi in charge_pos:
            system.part.add(pos=pi, q=qi)

        p3m = espressomd.electrostatics.P3M(
            prefactor=pref,
            accuracy=pw_error,
            check_neutrality=False,
            verbose=False
        )
        try:
            # Standard API binding for ELC (ESPResSo 4.x+)
            elc = espressomd.electrostatics.ELC(actor=p3m, gap_size=gap_val, maxPWerror=pw_error)
            system.electrostatics.solver = elc
        except TypeError:
            # Fallback for alternative/older binding schemes
            system.electrostatics.solver = p3m
            elc = espressomd.electrostatics.ELC(gap_size=gap_val, maxPWerror=pw_error)
            system.electrostatics.solver.add(elc)
            
        system.integrator.run(0)
        E = system.analysis.energy()["total"]
        system.electrostatics.clear()
        return E

    E_LT = calc_elc_energy(shifted(L_T), box_z_T, gap_T)
    E_Lpm1 = calc_elc_energy(shifted(L_neg1 + L_pos1), box_z_T, gap_T)
    E_L0 = calc_elc_energy(shifted(L0), box_z_T, gap_T)

    # Calculate E_near explicitly from subset energies
    E_near = 0.5 * (E_LT - E_Lpm1 + E_L0)

    # --- Part 3: Evaluate Far-Field Analytic Formulas for L_pm2 ---
    def L_pq(z, f_pq):
        return np.exp(-2 * np.pi * f_pq * z) / (1.0 - delta * np.exp(-4 * np.pi * lz * f_pq))

    def I_z(z):
        return (1.0 / (1.0 - delta)) * (z + 2 * lz * delta / (1.0 - delta))

    ux, uy = 1.0 / lx, 1.0 / ly
    K_cut = params.get("K_cut", 10)

    # Constant background integral terms (\xi)
    xi_L0_0 = sum([qi for qi, pi in L0]) if L0 else 0.0
    xi_L0_1 = sum([qi * pi[2] for qi, pi in L0]) if L0 else 0.0

    xi_L_minus2_0, xi_L_minus2_1 = 0.0, 0.0
    for qi, pi in L0_neg1:
        xi_L_minus2_0 += (qi / (1 - delta)) * (db * delta + delta)
        xi_L_minus2_1 += (qi / (1 - delta)) * (-db * delta * I_z(2*lz + pi[2]) - delta * I_z(2*lz - pi[2]))
    for qi, pi in L0_0 + L0_pos1:
        xi_L_minus2_0 += (qi / (1 - delta)) * (db + delta)
        xi_L_minus2_1 += (qi / (1 - delta)) * (-db * I_z(pi[2]) - delta * I_z(2*lz - pi[2]))

    xi_L_plus2_0, xi_L_plus2_1 = 0.0, 0.0
    for qi, pi in L0_pos1:
        xi_L_plus2_0 += (qi / (1 - delta)) * (dt * delta + delta)
        xi_L_plus2_1 += (qi / (1 - delta)) * (dt * delta * I_z(4*lz - pi[2]) + delta * I_z(2*lz + pi[2]))
    for qi, pi in L0_0 + L0_neg1:
        xi_L_plus2_0 += (qi / (1 - delta)) * (dt + delta)
        xi_L_plus2_1 += (qi / (1 - delta)) * (dt * I_z(2*lz - pi[2]) + delta * I_z(2*lz + pi[2]))

    # Cross-interaction summations
    sum_pq_plus2 = 0.0
    sum_pq_minus2 = 0.0

    for p in range(-K_cut, K_cut + 1):
        for q_idx in range(-K_cut, K_cut + 1):
            if p == 0 and q_idx == 0:
                continue
                
            f_pq = np.sqrt((p * ux)**2 + (q_idx * uy)**2)
            wp = 2 * np.pi * p * ux
            wq = 2 * np.pi * q_idx * uy

            cc0_p, sc0_p, cs0_p, ss0_p = 0.0, 0.0, 0.0, 0.0
            cc0_m, sc0_m, cs0_m, ss0_m = 0.0, 0.0, 0.0, 0.0
            
            for qi, pi in L0:
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0])
                cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                
                val_p = qi * np.exp(2 * np.pi * f_pq * pi[2])
                cc0_p += val_p * cp * cq; sc0_p += val_p * sp * cq; cs0_p += val_p * cp * sq; ss0_p += val_p * sp * sq
                
                val_m = qi * np.exp(-2 * np.pi * f_pq * pi[2])
                cc0_m += val_m * cp * cq; sc0_m += val_m * sp * cq; cs0_m += val_m * cp * sq; ss0_m += val_m * sp * sq

            cc_m2, sc_m2, cs_m2, ss_m2 = 0.0, 0.0, 0.0, 0.0
            for qi, pi in L0_neg1:
                val = qi * (db * delta * L_pq(2*lz + pi[2], f_pq) + delta * L_pq(2*lz - pi[2], f_pq))
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0]); cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_m2 += val * cp * cq; sc_m2 += val * sp * cq; cs_m2 += val * cp * sq; ss_m2 += val * sp * sq
            for qi, pi in L0_0 + L0_pos1:
                val = qi * (db * L_pq(pi[2], f_pq) + delta * L_pq(2*lz - pi[2], f_pq))
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0]); cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_m2 += val * cp * cq; sc_m2 += val * sp * cq; cs_m2 += val * cp * sq; ss_m2 += val * sp * sq

            cc_p2, sc_p2, cs_p2, ss_p2 = 0.0, 0.0, 0.0, 0.0
            for qi, pi in L0_pos1:
                val = qi * (dt * delta * L_pq(4*lz - pi[2], f_pq) + delta * L_pq(2*lz + pi[2], f_pq))
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0]); cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_p2 += val * cp * cq; sc_p2 += val * sp * cq; cs_p2 += val * cp * sq; ss_p2 += val * sp * sq
            for qi, pi in L0_0 + L0_neg1:
                val = qi * (dt * L_pq(2*lz - pi[2], f_pq) + delta * L_pq(2*lz + pi[2], f_pq))
                cp, sp = np.cos(wp * pi[0]), np.sin(wp * pi[0]); cq, sq = np.cos(wq * pi[1]), np.sin(wq * pi[1])
                cc_p2 += val * cp * cq; sc_p2 += val * sp * cq; cs_p2 += val * cp * sq; ss_p2 += val * sp * sq

            term_p2 = cc_p2 * cc0_p + sc_p2 * sc0_p + cs_p2 * cs0_p + ss_p2 * ss0_p
            sum_pq_plus2 += term_p2 / f_pq

            term_m2 = cc0_m * cc_m2 + sc0_m * sc_m2 + cs0_m * cs_m2 + ss0_m * ss_m2
            sum_pq_minus2 += term_m2 / f_pq

    # Combine analytic Fourier sums enforcing 1/2 multiplier 
    Phi_plus2_half = 0.5 * ux * uy * sum_pq_plus2 - np.pi * ux * uy * (xi_L_plus2_1 * xi_L0_0 - xi_L_plus2_0 * xi_L0_1)
    Phi_minus2_half = 0.5 * ux * uy * sum_pq_minus2 - np.pi * ux * uy * (xi_L0_1 * xi_L_minus2_0 - xi_L0_0 * xi_L_minus2_1)

    E_far = pref * (Phi_plus2_half + Phi_minus2_half)

    # --- Part 4: Cleanup & Aggregation ---
    E_total = E_near + E_far

    system.part.clear()
    system.box_l = [lx, ly, lz_full]
    for qi, pi in L0:
        system.part.add(pos=pi, q=qi)

    return {
        "E_total": E_total,
        "E_near": E_near,
        "E_near_p3m": E_near, 
        "E_near_corr": 0.0,
        "E_far": E_far,
    }