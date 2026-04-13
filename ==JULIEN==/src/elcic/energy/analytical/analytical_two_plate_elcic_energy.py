import numpy as np
"""
params = {
    "lx": 20.0,
    "ly": 20.0,
    ...
}
MULT=1, d=10s
MULT=2, d=45s
"""

def analytical_two_plate_elcic_energy(system, params, tol=1e-10):
    MULT = 1
    N_MAX = 100 * MULT
    M_MAX = 20 * MULT
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    slab_h = lz - params["gap_size"]

    def vectorized_pair_sum(pos_target, q_target, pos_sources, q_sources, is_self=False):
        """Calculates energy between target charges and a set of source charges including 2D images."""
        total_e = 0.0
        n_max = 1
        
        # Pre-calculate central interaction (n=0, n=0)
        dx = pos_target[:, 0][:, np.newaxis] - pos_sources[:, 0]
        dy = pos_target[:, 1][:, np.newaxis] - pos_sources[:, 1]
        dz = pos_target[:, 2][:, np.newaxis] - pos_sources[:, 2]
        dist_sq = dx**2 + dy**2 + dz**2
        
        # Mask self-interaction if target and sources are the same set
        if is_self:
            np.fill_diagonal(dist_sq, np.inf)
            
        dist = np.sqrt(dist_sq)
        # q_target (N,1) * q_sources (1, N) creates the (N,N) charge matrix
        total_e += np.sum((q_target[:, np.newaxis] * q_sources) / dist)

        while n_max <= N_MAX:
            # Generate the coordinates for the current shell only
            r = np.arange(-n_max, n_max + 1)
            # Create a grid of nx, ny
            nx, ny = np.meshgrid(r, r)
            # Filter for only the outer perimeter (the "shell")
            mask = (np.abs(nx) == n_max) | (np.abs(ny) == n_max)
            nx_shell = nx[mask] * lx
            ny_shell = ny[mask] * ly
            
            shell_e = 0.0
            # Broad-cast the shell offsets against the particle pairs
            # This is memory intensive but extremely fast
            for sx, sy in zip(nx_shell, ny_shell):
                d_shell_sq = (dx - sx)**2 + (dy - sy)**2 + dz**2
                shell_e += np.sum((q_target[:, np.newaxis] * q_sources) / np.sqrt(d_shell_sq))
            
            total_e += shell_e
            if n_max > 2 and abs(shell_e) < abs(total_e) * tol:
                break
            n_max += 1
        return total_e

    # 1. Real-Real interactions (Vectorized)
    # Using 0.5 because we calculate the full matrix (i,j and j,i)
    total_energy = 0.5 * vectorized_pair_sum(positions, charges, positions, charges, is_self=True)

    # 2. Image interactions
    m_max = 1
    while m_max <= M_MAX:
        m_energy = 0.0
        
        # Bottom-side image positions
        # q_img_b calculation
        qb = charges * (delta_b**m_max) * (delta_t**(m_max-1))
        zb = -positions[:, 2] - 2*(m_max-1)*slab_h
        pos_b = positions.copy()
        pos_b[:, 2] = zb
        
        # Top-side image positions
        qt = charges * (delta_t**m_max) * (delta_b**(m_max-1))
        zt = 2*m_max*slab_h - positions[:, 2]
        pos_t = positions.copy()
        pos_t[:, 2] = zt

        # Sum contributions
        m_energy += 0.5 * vectorized_pair_sum(positions, charges, pos_b, qb)
        m_energy += 0.5 * vectorized_pair_sum(positions, charges, pos_t, qt)

        total_energy += m_energy
        if abs(m_energy) < abs(total_energy) * tol:
            break
        m_max += 1

    return total_energy * prefactor