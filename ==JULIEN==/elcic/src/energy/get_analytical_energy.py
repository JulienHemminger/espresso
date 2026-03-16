def calculate_elcic_energy(params):
    """
    Computes the total electrostatic energy for a 2D+h system
    with two dielectric interfaces based on Tyagi et al. (2008).
    """
    box_l = params["box_l"]
    prefactor = params["prefactor"]
    dt = params["delta_mid_top"]
    db = params["delta_mid_bot"]
    delta = dt * db

    # Particle properties
    q = params["charges"]
    z = [params["p1_pos_z"], params["p1_pos_z"] + params["r_p1_p2"]]

    energy = 0.0
    # Interaction between the two real charges
    r_12 = abs(z[0] - z[1])
    energy += prefactor * q[0] * q[1] / r_12

    # Interaction with image charges (direct summation)
    # We sum over 'n' generations of reflections.
    # For delta < 1, this converges quickly.
    max_gen = 100

    for i in range(2):
        for j in range(2):
            qi, qj = q[i], q[j]
            zi, zj = z[i], z[j]

            # Sum over infinite image sequences defined in the paper
            # Lower dielectric sequences (Eq. 2.3 & 2.4)
            for n in range(max_gen):
                # Image at -(2*n*box_l + zj) with charge qj * (delta^n * db)
                pos_down1 = -(2 * n * box_l + zj)
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * db)) / abs(zi - pos_down1)
                )

                # Image at -(2*(n+1)*box_l - zj) with charge qj * delta^(n+1)
                if n < max_gen - 1:  # Avoid double counting or out of range
                    pos_down2 = -(2 * (n + 1) * box_l - zj)
                    energy += (
                        0.5
                        * prefactor
                        * qi
                        * (qj * delta ** (n + 1))
                        / abs(zi - pos_down2)
                    )

            # Upper dielectric sequences (Eq. 2.5 & 2.6)
            for n in range(max_gen):
                # Image at (2*(n+1)*box_l - zj) with charge qj * (delta^n * dt)
                pos_up1 = 2 * (n + 1) * box_l - zj
                energy += (
                    0.5 * prefactor * qi * (qj * (delta**n * dt)) / abs(zi - pos_up1)
                )

                # Image at (2*(n+1)*box_l + zj) with charge qj * delta^(n+1)
                pos_up2 = 2 * (n + 1) * box_l + zj
                energy += (
                    0.5 * prefactor * qi * (qj * delta ** (n + 1)) / abs(zi - pos_up2)
                )

    return energy
