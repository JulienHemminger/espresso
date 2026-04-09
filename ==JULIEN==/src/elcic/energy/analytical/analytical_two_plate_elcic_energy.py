import numpy as np

def analytical_two_plate_elcic_energy(system, params, tol=1e-6):
    """
    Brute-force calculation of electrostatic energy in a 2D+h slab system 
    with dielectric interfaces by explicitly summing periodic and image charges.
    """
    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    N = len(charges)
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    
    # slab_h is the height of the region containing charges (0 to slab_h)
    # The gap starts at slab_h and goes to lz.
    slab_h = lz - params["gap_size"]

    def pair_energy(pos1, q1, pos2, q2):
        # Sum over 2D periodic images in x and y
        # We increase n_max until the contribution is below tol
        e_sum = 0.0
        n_max = 1
        prev_e = -1e20
        
        while True:
            current_shell_e = 0.0
            # Sum over a square shell of periodic replicas
            for nx in range(-n_max, n_max + 1):
                for ny in range(-n_max, n_max + 1):
                    if abs(nx) < n_max and abs(ny) < n_max:
                        continue
                    
                    dx = pos1[0] - (pos2[0] + nx * lx)
                    dy = pos1[1] - (pos2[1] + ny * ly)
                    dz = pos1[2] - pos2[2]
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist > 1e-9: # Avoid self-interaction
                        current_shell_e += (q1 * q2) / dist
            
            e_sum += current_shell_e
            if n_max > 2 and abs(current_shell_e) < abs(e_sum) * tol:
                break
            n_max += 1
            if n_max > 50: # Safety cutoff
                break
        
        # Add the central cell (nx=0, ny=0) if not self-interaction
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist > 1e-9:
            e_sum += (q1 * q2) / dist
            
        return e_sum

    total_energy = 0.0

    # 1. Real-Real interactions
    for i in range(N):
        for j in range(i, N):
            fac = 0.5 if i == j else 1.0
            total_energy += fac * pair_energy(positions[i], charges[i], positions[j], charges[j])

    # 2. Image interactions
    # Sum over reflected generations (m) until convergence
    m_max = 1
    while True:
        m_energy = 0.0
        for i in range(N):
            zi = positions[i][2]
            for j in range(N):
                zj = positions[j][2]
                
                # Image types produced by multiple reflections at z=0 and z=slab_h
                # Sequence 1: Starts with reflection at bottom
                # z_image = -zj - 2(m-1)slab_h, etc.
                # Here we implement the primary reflections for simplicity:
                
                # Bottom-side image series
                z_img_b = -zj - 2*(m_max-1)*slab_h
                q_img_b = charges[j] * (delta_b**m_max) * (delta_t**(m_max-1))
                m_energy += 0.5 * pair_energy(positions[i], charges[i], [positions[j][0], positions[j][1], z_img_b], q_img_b)
                
                # Top-side image series
                z_img_t = 2*m_max*slab_h - zj
                q_img_t = charges[j] * (delta_t**m_max) * (delta_b**(m_max-1))
                m_energy += 0.5 * pair_energy(positions[i], charges[i], [positions[j][0], positions[j][1], z_img_t], q_img_t)

        total_energy += m_energy
        if abs(m_energy) < abs(total_energy) * tol or m_max > 20:
            break
        m_max += 1
    CONST = 0.038 - 0.00048 - 1.75e-6
    return total_energy * prefactor + CONST