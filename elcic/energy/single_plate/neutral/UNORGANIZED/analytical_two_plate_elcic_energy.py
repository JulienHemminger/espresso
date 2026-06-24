import numpy as np
import espressomd
import espressomd.electrostatics

"""Example parameters
params = {
    "lx": 11,
    "ly": 14,
    "lz": 19,
    "gap_size": 6,
    "prefactor": 1.0,
    "delta_mid_top": -0.6,
    "delta_mid_bot": 1.0,
    "pw_error": 1e-8,
    "charges": [+1, -1],
}
params["positions"] = [
    np.array([5.4, 2.2, 11.1]),
    np.array([3.7, 6.2, 1.1]),
]

"""

def ewald_two_plate_elcic_energy(system:espressomd.System, gap_size:float, pw_error:float, prefactor:float, delta_mid_bot:float, delta_mid_top:float):
    """
    Calculate total electrostatic energy of a 2d+h slab system with dielectric interfaces.
    
    Parameters:
    -----------
    system : espressomd.System
        The simulation system
    gap_size : float
        Gap size (λ) for layer separation
    pw_error : float
        accuracy target
    prefactor : float
        Electrostatic prefactor
    delta_mid_bot : float
        Dielectric contrast factor for bottom interface: (εm - εb)/(εm + εb)
    delta_mid_top : float
        Dielectric contrast factor for top interface: (εm - εt)/(εm + εt)
    
    Returns:
    --------
    float: Energy
    """
    lx, ly, lz = system.box_l
    particles = system.part.all()
    qs, (xs, ys, zs) = particles.q, particles.pos.T
   

    return 0