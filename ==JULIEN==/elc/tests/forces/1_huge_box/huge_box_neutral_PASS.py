import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.energy.get_elc_energy import get_elc_energy, get_elc_energy_contribs
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
import matplotlib.pyplot as plt
from scipy.stats import linregress

from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from elc.src.forces.get_legacy_forces import get_legacy_forces
from elc.src.common.set_utils import are_sets_equal
from elc.src.forces.get_elc_forces import get_elc_forces
import numpy as np

def get_analytical_forces(system):
    # 1. Initialize a list of zero vectors for each particle
    particles = list(system.part.all())
    n = len(particles)
    # Using a dictionary or list to store forces mapped to indices
    forces = [np.zeros(3) for _ in range(n)]
    
    # ke is the Coulomb constant; adjust based on your simulation units
    ke = 1.0 

    # 2. Double loop for pair-wise interactions
    for i in range(n):
        for j in range(i + 1, n):
            p1 = particles[i]
            p2 = particles[j]
            
            # Distance vector and magnitude
            r_vec = p1.pos - p2.pos
            dist_sq = np.sum(r_vec**2)
            dist = np.sqrt(dist_sq)
            
            if dist == 0:
                continue # Avoid division by zero for overlapping particles
                
            # Coulomb's Law calculation
            # F = ke * (q1 * q2 / r^2) * (r_vec / r)
            force_mag = ke * (p1.q * p2.q) / dist_sq
            force_vec = force_mag * (r_vec / dist)
            
            # 3. Accumulate forces (Action = -Reaction)
            forces[i] += force_vec
            forces[j] -= force_vec
            
    return forces

def test_accuracy_convergence():
    l_x, l_y = 200.0, 200.0
    l_z = 10.0
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    
    pw_err = 1e-4
    gap_size = 1.0
    
    pos1, pos2, pos3 = get_rdm_constrained_points_np(l_x, l_y, l_z-gap_size-1e-3, 3, max_distance = 10)
   
    
    system.part.clear()
    system.part.add(pos=pos1, q=-1)
    system.part.add(pos=pos2, q=-1)
    system.part.add(pos=pos3, q=+2)
    
    
    
    analytical_forces = get_analytical_forces(system)
    
    legacy_forces = get_legacy_forces(system, gap_size, pw_err)
    #legacy_forces = get_elc_forces(system, gap_size, pw_err) ## nur tol=1e3*pw_err
    
    for f in legacy_forces:
        print(f"{str(f)}")
        
    print("=========")
    for f in analytical_forces:
        print(f"{str(f)}")    
    assert are_sets_equal(analytical_forces, legacy_forces, tol=1e2*pw_err)
        
        