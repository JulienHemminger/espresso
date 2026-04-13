"""
params = {
    "lx": 20.0,
    "ly": 20.0,
    ...
}
MULT=1, d=10s
MULT=2, d=45s
"""
import numpy as np

def analytical_two_plate_elcic_energy(system, params, tol=1e-6, N_CUTOFF=5_000, M_CUTOFF=20):
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    slab_h = lz - params["gap_size"]
    
    PRINT_STRIDE = N_CUTOFF/5 # Only print every 10 steps to avoid clutter

    def vectorized_pair_sum(pos_target, q_target, pos_sources, q_sources, is_self=False, label="Real"):
        """Calculates energy with convergence monitoring."""
        total_e = 0.0
        n_max = 1
        
        dx = pos_target[:, 0][:, np.newaxis] - pos_sources[:, 0]
        dy = pos_target[:, 1][:, np.newaxis] - pos_sources[:, 1]
        dz = pos_target[:, 2][:, np.newaxis] - pos_sources[:, 2]
        dist_sq = dx**2 + dy**2 + dz**2
        
        if is_self:
            np.fill_diagonal(dist_sq, np.inf)
            
        dist = np.sqrt(dist_sq)
        total_e += np.sum((q_target[:, np.newaxis] * q_sources) / dist)

        while n_max <= N_CUTOFF:
            r = np.arange(-n_max, n_max + 1)
            nx, ny = np.meshgrid(r, r)
            mask = (np.abs(nx) == n_max) | (np.abs(ny) == n_max)
            nx_shell = nx[mask] * lx
            ny_shell = ny[mask] * ly
            
            shell_e = 0.0
            for sx, sy in zip(nx_shell, ny_shell):
                d_shell_sq = (dx - sx)**2 + (dy - sy)**2 + dz**2
                shell_e += np.sum((q_target[:, np.newaxis] * q_sources) / np.sqrt(d_shell_sq))
            
            prev_total = total_e
            total_e += shell_e
            
            # Convergence tracking
            rel_diff = abs(shell_e / total_e) if total_e != 0 else 0
            
            if n_max % PRINT_STRIDE == 0:
                print(f"  [{label} N={n_max}] Energy: {total_e:.6e} | Rel Change: {rel_diff:.2e}")

            if n_max > 2 and rel_diff < tol:
                print(f"  --> {label} converged at N={n_max} (Rel Change: {rel_diff:.2e})")
                return total_e
            
            n_max += 1
            
        print(f"  !! WARNING: {label} reached N_CUTOFF ({N_CUTOFF}) without converging.")
        return total_e

    print(f"--- Starting Convergence Check (Tol: {tol}) ---")
    
    # 1. Real-Real interactions
    total_energy = 0.5 * vectorized_pair_sum(positions, charges, positions, charges, is_self=True, label="Periodic Shells")

    # 2. Image interactions
    print(f"\n--- Starting Image Method (M_MAX: {M_CUTOFF}) ---")
    m_max = 1
    while m_max <= M_CUTOFF:
        m_energy = 0.0
        
        qb = charges * (delta_b**m_max) * (delta_t**(m_max-1))
        zb = -positions[:, 2] - 2*(m_max-1)*slab_h
        pos_b = positions.copy()
        pos_b[:, 2] = zb
        
        qt = charges * (delta_t**m_max) * (delta_b**(m_max-1))
        zt = 2*m_max*slab_h - positions[:, 2]
        pos_t = positions.copy()
        pos_t[:, 2] = zt

        # We pass a high tol here to avoid nested printing spam, or keep it consistent
        m_energy += 0.5 * vectorized_pair_sum(positions, charges, pos_b, qb, label=f"M={m_max} Bot")
        m_energy += 0.5 * vectorized_pair_sum(positions, charges, pos_t, qt, label=f"M={m_max} Top")

        prev_total = total_energy
        total_energy += m_energy
        rel_m_diff = abs(m_energy / total_energy) if total_energy != 0 else 0
        
        print(f"Iteration M={m_max}: Energy contribution: {m_energy:.6e} | Total: {total_energy:.6e}")

        if abs(m_energy) < abs(total_energy) * tol:
            print(f"--> Image method converged at M={m_max}")
            break
        
        if m_max == M_CUTOFF:
             print(f"!! WARNING: Image method reached M_CUTOFF ({M_CUTOFF}) without converging.")
             
        m_max += 1

    final_val = total_energy * prefactor
    print(f"\nFinal Calculated Energy: {final_val:.8f}")
    return final_val