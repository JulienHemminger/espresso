import espressomd
import espressomd.electrostatics
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc, erf

def get_ewald_energy_2d(system, n_max=100):
    positions = system.part.all().pos  # Shape (N, 3)
    charges = system.part.all().q      # Shape (N,)

    # 2. Get box dimensions (assuming a rectangular box)
    lx = system.box_l[0]
    ly = system.box_l[1]
        
    eta=None
    n_real=n_max
    n_recip=n_max
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)
    A = lx * ly

    if eta is None:
        # Balance real/reciprocal convergence
        eta = np.sqrt(np.pi) / min(lx, ly)

    # Pair separation vectors: dr[a, b] = pos[a] - pos[b]
    dr = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    qq = q[:, None] * q[None, :]                     # (N, N)

    # ---- Real-space sum ----
    E_real = 0.0
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            R = np.array([nx * lx, ny * ly, 0.0])
            rvec = dr + R                             # (N, N, 3)
            dist = np.linalg.norm(rvec, axis=2)       # (N, N)

            if nx == 0 and ny == 0:
                np.fill_diagonal(dist, np.inf)        # exclude self

            contrib = qq * erfc(eta * dist) / dist
            E_real += np.sum(contrib)
    E_real *= 0.5

    # ---- Reciprocal-space sum (G != 0) ----
    gx0 = 2.0 * np.pi / lx
    gy0 = 2.0 * np.pi / ly
    drho = dr[:, :, :2]                               # in-plane (N, N, 2)
    dz = dr[:, :, 2]                                  # z-separation (N, N)

    E_recip = 0.0
    for mx in range(-n_recip, n_recip + 1):
        for my in range(-n_recip, n_recip + 1):
            if mx == 0 and my == 0:
                continue
            Gx = mx * gx0
            Gy = my * gy0
            G = np.sqrt(Gx**2 + Gy**2)

            phase = drho[:, :, 0] * Gx + drho[:, :, 1] * Gy  # (N, N)

            # h(G, dz) = exp(G*dz)*erfc(G/(2*eta) + eta*dz)
            #           + exp(-G*dz)*erfc(G/(2*eta) - eta*dz)
            arg_p = G / (2.0 * eta) + eta * dz
            arg_m = G / (2.0 * eta) - eta * dz
            h = np.exp(G * dz) * erfc(arg_p) + np.exp(-G * dz) * erfc(arg_m)

            E_recip += np.sum(qq * (np.pi / G) * h * np.cos(phase))

    E_recip /= (2.0 * A)

    # ---- Self-energy correction ----
    E_self = -(eta / np.sqrt(np.pi)) * np.sum(q ** 2)

    # ---- G = 0 term ----
    # Limit for |dz| -> 0:  |dz|*erf(eta*|dz|) + exp(-(eta*dz)^2)/(eta*sqrt(pi))
    #                      -> 1/(eta*sqrt(pi))
    adz = np.abs(dz)
    g0_terms = np.where(
        adz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        adz * erf(eta * adz) + np.exp(-(eta * adz) ** 2) / (eta * np.sqrt(np.pi)),
    )
    E_G0 = -np.pi / A * np.sum(qq * g0_terms)

    E_total = E_real + E_recip + E_self + E_G0
    return E_total

l_xy = 10.0
l_z = 3.0
gap_size= 1.0
system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

z_range = np.linspace(0, l_z - gap_size -1e-3, num=10)
n_int = 100
ewald_energies = []

for z in z_range:
    system.part.clear()
    
    system.part.add(pos=[3.0, 5.0, z], q=+2.0)
    system.part.add(pos=[7.0, 5.0, z], q=-1.0)
    
    e_ewald = get_ewald_energy_2d(system, n_int)
    ewald_energies.append(e_ewald)


plt.figure(figsize=(10, 6))

plt.plot(z_range, ewald_energies, '-', color='tab:red', linewidth=2, label='Ewald 2D Energy ($n_{max}=100$)')

plt.xlabel('Dipole Height ($z$)')
plt.ylabel('Interaction Energy')
plt.title('2D Ewald Energy vs. Vertical Position (z)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()