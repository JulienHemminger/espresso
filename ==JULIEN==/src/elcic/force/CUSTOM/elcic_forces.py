import espressomd
import espressomd.electrostatics
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy # def get_elcic_energy(system, params: dict):

import numpy as np
import copy

def get_elcic_forces(system, params: dict, delta=1e-4):
    """
    Computes numerical gradient of get_elcic_energy to find forces.
    
    Args:
        system: EspressoMD system instance.
        params: Dictionary containing 'positions' and system config.
        delta: Small displacement for finite difference.
    """
    num_particles = len(params["positions"])
    forces = [np.zeros(3) for _ in range(num_particles)]
    
    # Store original positions to restore them later
    original_positions = [pos.copy() for pos in params["positions"]]
    
    for i in range(num_particles):
        for k in range(3):  # For x, y, z dimensions
            # 1. Compute U(r + delta)
            params["positions"][i][k] += delta
            energy_plus = get_elcic_energy(system, params)["e_total"]
            
            # 2. Compute U(r - delta)
            params["positions"][i][k] -= 2 * delta
            energy_minus = get_elcic_energy(system, params)["e_total"]
            
            # 3. Calculate central difference gradient
            gradient = (energy_plus - energy_minus) / (2 * delta)
            
            # 4. Force is negative gradient
            forces[i][k] = -gradient
            
            # Restore position for next iteration
            params["positions"][i][k] += delta
            
    # Ensure positions are restored to their initial state
    params["positions"] = original_positions
    
    return forces