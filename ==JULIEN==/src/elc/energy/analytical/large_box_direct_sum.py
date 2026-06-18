import numpy as np


def calculate_energy(system, prefactor=1.0):
    """
    Calculates the electrostatic potential energy using scalar math 
    and nested loops for clarity.
    """
    pos = system.part.all().pos
    q = system.part.all().q
    n_particles = len(q)
    
    total_energy = 0.0
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            p1 = pos[i]
            p2 = pos[j]
            
            # Calculate scalar distance: sqrt(dx^2 + dy^2 + dz^2)
            r = np.linalg.norm(p1 - p2)
            
            # Add interaction energy for this pair
            total_energy += (q[i] * q[j]) / r
            
    return prefactor * total_energy


import espressomd
import espressomd.electrostatics
import numpy as np


system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"




params = {
    "lx": 20.0,
    "ly": 20.0,
    "lz": 10.0,
    "gap_size": 5.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1]), np.array([4, 2, 1])],
    "pw_error": 1e-8,
}



system.electrostatics.clear()
system.part.clear()
system.box_l = [params["lx"], params["ly"], params["lz"]]

for i in range(len(params["charges"])):
    system.part.add(pos=params["positions"][i], q=params["charges"][i])


from src.elc.energy.legacy_elc_energy import get_legacy_energy

analytical_energy = float(calculate_energy(system))
print(f"{analytical_energy=}")

legacy_energy = get_legacy_energy(system, params)["total"]
print(f"{legacy_energy=}")

"""
lxy = 10
* analytical_energy = 0.5943491939145144
* legacy_energy = 0.42799823487395017

legacy
* runs for lxy = 10, 20, 30, 60
* doesnt run for lxy = 100

lxy = 100
* analytical_energy=0.5943491939145144
* legacy_energy timed out
"""