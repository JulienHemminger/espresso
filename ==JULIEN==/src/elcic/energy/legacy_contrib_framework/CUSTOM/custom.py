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
    eps = params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db = params["delta_mid_bot"]
    dt = params["delta_mid_top"]
    
    lz = lz_full - gap
    parts = system.part.all()
    charges, positions = parts.q.copy(), parts.pos.copy()
    
    # Splitting parameter lambda
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)
    
    # Evaluate effective Delta constraint
    delta = db * dt
    if abs(delta - 1.0) < 1e-7:
        delta = 1.0 - 1e-7 if delta > 0 else -1.0 + 1e-7

    # ---------------------------------------------------------
    # 1. FAR-FIELD COMPUTATION (E_far) 
    # Interaction of original charges with distant images L_+2 and L_-2
    # ---------------------------------------------------------
    def compute_e_far():
        if len(charges) == 0:
            return 0.0
            
        ux, uy = 1.0 / lx, 1.0 / ly
        
        # Determine frequency cutoff based on pw_error
        R_cut = int(np.ceil(max(lx, ly) / (2 * np.pi * lambda_) * (-np.log(eps))))
        R_cut = max(min(R_cut, 30), 1) 
        
        z = positions[:, 2]
        idx_m1 = np.where(z < lambda_)[0]
        idx_0  = np.where((z >= lambda_) & (z <= lz - lambda_))[0]
        idx_p1 = np.where(z > lz - lambda_)[0]
        
        xi_L0_0 = np.sum(charges)
        xi_L0_1 = np.sum(charges * z)
        
        def I_z(zi):
            return (1.0 / (1.0 - delta)) * (zi + 2.0 * lz * delta / (1.0 - delta))
            
        # Dipole moment sums for L_-2
        xi_Lm2_0, xi_Lm2_1 = 0.0, 0.0
        for i in idx_m1:
            q = charges[i]
            xi_Lm2_0 += q / (1 - delta) * (db * delta + delta)
            xi_Lm2_1 += q / (1 - delta) * (-db * delta * I_z(2*lz + z[i]) - delta * I_z(2*lz - z[i]))
        for i in np.concatenate((idx_0, idx_p1)):
            q = charges[i]
            xi_Lm2_0 += q / (1 - delta) * (db + delta)
            xi_Lm2_1 += q / (1 - delta) * (-db * I_z(z[i]) - delta * I_z(2*lz - z[i]))
            
        # Dipole moment sums for L_+2
        xi_Lp2_0, xi_Lp2_1 = 0.0, 0.0
        for i in idx_p1:
            q = charges[i]
            xi_Lp2_0 += q / (1 - delta) * (dt * delta + delta)
            xi_Lp2_1 += q / (1 - delta) * (dt * delta * I_z(4*lz - z[i]) + delta * I_z(2*lz + z[i]))
        for i in np.concatenate((idx_0, idx_m1)):
            q = charges[i]
            xi_Lp2_0 += q / (1 - delta) * (dt + delta)
            xi_Lp2_1 += q / (1 - delta) * (dt * I_z(2*lz - z[i]) + delta * I_z(2*lz + z[i]))
            
        e_far_sum = 0.0
        
        for p in range(-R_cut, R_cut + 1):
            for q_mode in range(-R_cut, R_cut + 1):
                if (p == 0 and q_mode == 0) or (p**2 + q_mode**2 > R_cut**2):
                    continue
                    
                fpq = np.sqrt((ux * p)**2 + (uy * q_mode)**2)
                wp, wq = 2 * np.pi * ux * p, 2 * np.pi * uy * q_mode
                
                def L_pq(zi):
                    return np.exp(-2 * np.pi * fpq * zi) / (1 - delta * np.exp(-4 * np.pi * lz * fpq))
                    
                # Evaluate trig combinations
                c_wp, s_wp = np.cos(wp * positions[:, 0]), np.sin(wp * positions[:, 0])
                c_wq, s_wq = np.cos(wq * positions[:, 1]), np.sin(wq * positions[:, 1])
                exp_plus, exp_minus = np.exp(2 * np.pi * fpq * z), np.exp(-2 * np.pi * fpq * z)
                
                chi_L0_m_cc = np.sum(charges * exp_minus * c_wp * c_wq)
                chi_L0_m_sc = np.sum(charges * exp_minus * s_wp * c_wq)
                chi_L0_m_cs = np.sum(charges * exp_minus * c_wp * s_wq)
                chi_L0_m_ss = np.sum(charges * exp_minus * s_wp * s_wq)
                
                chi_L0_p_cc = np.sum(charges * exp_plus * c_wp * c_wq)
                chi_L0_p_sc = np.sum(charges * exp_plus * s_wp * c_wq)
                chi_L0_p_cs = np.sum(charges * exp_plus * c_wp * s_wq)
                chi_L0_p_ss = np.sum(charges * exp_plus * s_wp * s_wq)
                
                # Chi coefficients for L_-2
                A = np.zeros_like(z)
                for i in idx_m1:
                    A[i] = db * delta * L_pq(2*lz + z[i]) + delta * L_pq(2*lz - z[i])
                for i in np.concatenate((idx_0, idx_p1)):
                    A[i] = db * L_pq(z[i]) + delta * L_pq(2*lz - z[i])
                    
                chi_Lm2_p_cc = np.sum(charges * A * c_wp * c_wq)
                chi_Lm2_p_sc = np.sum(charges * A * s_wp * c_wq)
                chi_Lm2_p_cs = np.sum(charges * A * c_wp * s_wq)
                chi_Lm2_p_ss = np.sum(charges * A * s_wp * s_wq)
                
                # Chi coefficients for L_+2
                B = np.zeros_like(z)
                for i in idx_p1:
                    B[i] = dt * delta * L_pq(4*lz - z[i]) + delta * L_pq(2*lz + z[i])
                for i in np.concatenate((idx_0, idx_m1)):
                    B[i] = dt * L_pq(2*lz - z[i]) + delta * L_pq(2*lz + z[i])
                    
                chi_Lp2_m_cc = np.sum(charges * B * c_wp * c_wq)
                chi_Lp2_m_sc = np.sum(charges * B * s_wp * c_wq)
                chi_Lp2_m_cs = np.sum(charges * B * c_wp * s_wq)
                chi_Lp2_m_ss = np.sum(charges * B * s_wp * s_wq)
                
                # Interaction combinations
                sum_m2 = chi_L0_m_cc * chi_Lm2_p_cc + chi_L0_m_sc * chi_Lm2_p_sc + \
                         chi_L0_m_cs * chi_Lm2_p_cs + chi_L0_m_ss * chi_Lm2_p_ss
                
                sum_p2 = chi_Lp2_m_cc * chi_L0_p_cc + chi_Lp2_m_sc * chi_L0_p_sc + \
                         chi_Lp2_m_cs * chi_L0_p_cs + chi_Lp2_m_ss * chi_L0_p_ss
                         
                e_far_sum += (1.0 / fpq) * (sum_m2 + sum_p2)
                
        e_far_sum *= (ux * uy)
        
        dipole_m2 = -2 * np.pi * ux * uy * (xi_L0_1 * xi_Lm2_0 - xi_L0_0 * xi_Lm2_1)
        dipole_p2 = -2 * np.pi * ux * uy * (xi_Lp2_1 * xi_L0_0 - xi_Lp2_0 * xi_L0_1)
        
        return pref * (e_far_sum + dipole_m2 + dipole_p2)

    e_far = compute_e_far()

    # ---------------------------------------------------------
    # 2. NEAR-FIELD COMPUTATION (E_near)
    # Using the native ESPResSo P3M + ELC logic on shifted configurations
    # ---------------------------------------------------------
    
    def get_system_energy(q_arr, pos_arr, box_lz, use_elc):
        if len(q_arr) == 0: 
            return 0.0
            
        system.electrostatics.solver = None
        if hasattr(system.electrostatics, 'extension'):
            system.electrostatics.extension = None
            
        system.part.clear()
        system.box_l = [lx, ly, box_lz]
        system.part.add(pos=pos_arr, q=q_arr)
        
        p3m = espressomd.electrostatics.P3M(
            prefactor=pref, accuracy=eps, check_neutrality=False, verbose=False
        )
        
        if use_elc:
            elc = espressomd.electrostatics.ELC(actor=p3m, gap_size=gap, maxPWerror=eps)
            system.electrostatics.solver = elc
        else:
            system.electrostatics.solver = p3m
            
        system.integrator.run(0)
        energy = system.analysis.energy()["total"]
        return energy

    # Establish charges and geometric shifts to handle the lambda gap cleanly
    z = positions[:, 2] if len(positions) > 0 else np.empty(0)
    
    idx_m1 = np.where(z < lambda_)[0]
    idx_p1 = np.where(z > lz - lambda_)[0]
    
    pos_0 = positions.copy()
    if len(pos_0) > 0:
        pos_0[:, 2] += lambda_
    q_0 = charges.copy()
    
    pos_m1 = positions[idx_m1].copy()
    if len(idx_m1) > 0:
        pos_m1[:, 2] = -z[idx_m1] + lambda_
    q_m1 = charges[idx_m1] * db
    
    pos_p1 = positions[idx_p1].copy()
    if len(idx_p1) > 0:
        pos_p1[:, 2] = 2 * lz - z[idx_p1] + lambda_
    q_p1 = charges[idx_p1] * dt
    
    pos_pm1 = np.vstack([pos_m1, pos_p1]) if (len(pos_m1) or len(pos_p1)) else np.empty((0, 3))
    q_pm1 = np.concatenate([q_m1, q_p1]) if (len(q_m1) or len(q_p1)) else np.empty(0)
    
    pos_T = np.vstack([pos_0, pos_pm1]) if len(pos_pm1) else pos_0
    q_T = np.concatenate([q_0, q_pm1]) if len(q_pm1) else q_0

    # Execute system passes
    box_lz_elc = lz + 2 * lambda_ + gap
    
    e_near_3d = get_system_energy(charges, positions, lz_full, use_elc=False)
    
    E_T   = get_system_energy(q_T, pos_T, box_lz_elc, use_elc=True)
    E_pm1 = get_system_energy(q_pm1, pos_pm1, box_lz_elc, use_elc=True)
    E_0   = get_system_energy(q_0, pos_0, box_lz_elc, use_elc=True)

    e_near = 0.5 * (E_T - E_pm1 + E_0)
    e_near_corr = e_near - e_near_3d
    e_total = e_near + e_far

    # Restore the original singleton system state
    system.electrostatics.solver = None
    if hasattr(system.electrostatics, 'extension'):
        system.electrostatics.extension = None
        
    system.part.clear()
    system.box_l = box
    if len(charges) > 0:
        system.part.add(pos=positions, q=charges)

    return {
        "E_total": e_total,
        "E_near": e_near,
        "E_near_p3m": e_near_3d,
        "E_near_corr": e_near_corr,
        "E_far": e_far,
    }