import numpy as np
import espressomd
import espressomd.electrostatics


def get_elc_forces(system, gap_size, pw_err) -> list[np.ndarray]:
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    f_3d = [p.f for p in system.part.all()]
    
    # todo: elc correction
    return f_3d
    