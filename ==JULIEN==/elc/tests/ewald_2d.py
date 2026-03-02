import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc, erf


def ewald_energy_2d(positions, charges, dx, dy, eta=None, n_real=8, n_recip=12):
    print("a")
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)
    A = dx * dy

    if eta is None:
        # Balance real/reciprocal convergence
        eta = np.sqrt(np.pi) / min(dx, dy)

    # Pair separation vectors: dr[a, b] = pos[a] - pos[b]
    dr = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    qq = q[:, None] * q[None, :]                     # (N, N)

    # ---- Real-space sum ----
    E_real = 0.0
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            R = np.array([nx * dx, ny * dy, 0.0])
            rvec = dr + R                             # (N, N, 3)
            dist = np.linalg.norm(rvec, axis=2)       # (N, N)

            if nx == 0 and ny == 0:
                np.fill_diagonal(dist, np.inf)        # exclude self

            contrib = qq * erfc(eta * dist) / dist
            E_real += np.sum(contrib)
    E_real *= 0.5

    # ---- Reciprocal-space sum (G != 0) ----
    gx0 = 2.0 * np.pi / dx
    gy0 = 2.0 * np.pi / dy
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


# Parameters for a simple test system
positions = np.array([[0.2, 0.2, 0.0], [0.7, 0.7, 0.0]])
charges = np.array([1.0, -1.0])
dx, dy = 1.0, 1.0

# Calculate energy for various n_max
# Using logspace to get 10 values from 10^2 (100) to 10^4 (10,000)
n_max_values = np.logspace(2, 3, num=2, dtype=int)
energies = [ewald_energy_2d(positions, charges, dx, dy, n_real=n, n_recip=n)[0] for n in n_max_values]

# Plotting
plt.plot(n_max_values, energies, marker='o', linestyle='-', color='b')
plt.xscale('log')  # Apply log scale to the x-axis
plt.xlabel('$n_{max}$')
plt.ylabel('Energy')
plt.title('Convergence of Direct Sum Energy vs. $n_{max}$')
plt.grid(True)
plt.savefig('energy_convergence.png')

# Print values to see trends
print(list(zip(n_max_values, energies)))