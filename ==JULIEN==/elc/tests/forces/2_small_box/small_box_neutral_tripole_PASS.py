from elc.src.common.get_positions import get_rdm_constrained_points_np
from elc.src.forces.get_elc_forces import get_elc_forces
from elc.src.forces.third_party.get_ewald_forces_2d import get_ewald_forces_2d
import espressomd
import espressomd.electrostatics
import numpy as np

def test_accuracy_convergence():
    l_x, l_y = 10.0, 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    
    pw_err = 1e-6
    gap_size = 1.0
    
    pos1, pos2, pos3 = get_rdm_constrained_points_np(l_x, l_y, l_z-gap_size-1e-3, 3, max_distance = 10)
   
    
    system.part.clear()
    system.part.add(pos=pos1, q=-1)
    system.part.add(pos=pos2, q=-1)
    system.part.add(pos=pos3, q=+2)
    
    
    
    analytical_forces = get_ewald_forces_2d(system)
    
    #legacy_forces = get_legacy_forces(system, gap_size, pw_err)
    legacy_forces = get_elc_forces(system, gap_size, pw_err) ## nur tol=1e3*pw_err
    
    for f in analytical_forces:
        print(f"{str(f)}")    
    print("=========")
    for f in legacy_forces:
        print(f"{str(f)}")
        
        
    #assert are_sets_equal(analytical_forces, legacy_forces, tol=1e3*pw_err)
        