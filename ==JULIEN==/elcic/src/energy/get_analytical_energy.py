import numpy as np


def calculate_elcic_energy(params):
    """
    Computes total electrostatic energy for an N-particle 2D+h system
    using image charge summation (Tyagi et al. 2008).
    """
    lz = params["lz"]
    prefactor = params["prefactor"]
    dt = params["delta_mid_top"]
    db = params["delta_mid_bot"]
    delta = dt * db

    q = np.array(params["charges"])
    pos = np.array(params["positions"])
    z = pos[:, 2]  # Extract only z-coordinates for image interactions
    n_part = len(q)

    energy = 0.0

    # 1. Direct Real-Real Interaction (Coulomb)
    # Using a double loop for clarity; can be vectorized for performance
    for i in range(n_part):
        for j in range(i + 1, n_part):
            r = np.linalg.norm(pos[i] - pos[j])
            energy += prefactor * q[i] * q[j] / r

    # 2. Image Charge Interactions
    # We sum over 'max_gen' reflections across the top and bottom interfaces
    max_gen = 100

    for i in range(n_part):
        for j in range(n_part):
            qi, qj = q[i], q[j]
            zi, zj = z[i], z[j]

            for n in range(max_gen):
                # --- Lower Interface Reflections ---
                # Eq 2.3: Image at -(2*n*Lz + zj)
                pos_down1 = -(2 * n * lz + zj)
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * db)) / abs(zi - pos_down1)
                )

                # Eq 2.4: Image at -(2*(n+1)*Lz - zj)
                if n < max_gen - 1:
                    pos_down2 = -(2 * (n + 1) * lz - zj)
                    energy += (
                        0.5
                        * prefactor
                        * qi
                        * (qj * delta ** (n + 1))
                        / abs(zi - pos_down2)
                    )

                # --- Upper Interface Reflections ---
                # Eq 2.5: Image at (2*(n+1)*Lz - zj)
                pos_up1 = 2 * (n + 1) * lz - zj
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * dt)) / abs(zi - pos_up1)
                )

                # Eq 2.6: Image at (2*(n+1)*Lz + zj)
                pos_up2 = 2 * (n + 1) * lz + zj
                energy += (
                    0.5 * prefactor * qi * (qj * delta ** (n + 1)) / abs(zi - pos_up2)
                )

    return energy
