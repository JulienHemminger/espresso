import numpy as np
import espressomd
import espressomd.electrostatics

def analytical_two_plate_elcic_energy(system, params, k_max=10, n_max=10):
    """
    n_max: the cutoff number of periodic boundary condition (PBC) "clones" of the central slab system in x and y direction
        * the total number of slabs per direction is n_max(left) + 1(center) + n_max(right)
        * in total theres (n_max + 1 + n_max)^2 slabs

    k_max: the cutoff number for reflections of image charges.
        * since "delta_mid_top" and "delta_mid_bot" are -1.0, there are infinite reflections back and forth between those two dielectric interfaces.
        * k_max is a cutoff, that cuts off this infinite sum to return a finite value


    Example for parameters:
        system: espressomd.System
        params = {
            "lx": 9.0,
            "ly": 12.0,
            "lz": 19.0,
            "gap_size": 15.0,
            "prefactor": 1.0,
            "delta_mid_top": -1.0,
            "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
            "charges": [+1, -1],
            'pw_error': 1e-8,
            "positions": [np.array([2, 5, 0]), np.array([8, 3, 0])],
            "title": "Dual Plates, Both Metallic, Neutral"
        }
    """


    positions = np.array([p.pos for p in system.part])
    charges = np.array([p.q for p in system.part])
    lz, lx, ly = params["lz"], params["lx"], params["ly"]
    prefactor = params["prefactor"]
    delta_b, delta_t = params["delta_mid_bot"], params["delta_mid_top"]
    pw_error = params["pw_error"]
    

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )

    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]

    return e_3d

