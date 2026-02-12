# %%
import sys
import os
# To allow me to write/run juptyer notebooks from VScode.
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np

pw_error = 1e-6
gap_size = 2.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, mesh=[124, 124, 120])
p3m1 = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, mesh=[124, 124, 120])

def get_elc_forces(actor, gap_size, pw_error, system):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error
    )
    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return p1.f

def get_p3m_forces(actor, gap_size, pw_error, system):
    system.electrostatics.solver = actor
    #system.integrator.run(0)
    return p1.f

l_xy = 100.0 # keep l_xy <= 200
l_z = 12.0
system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4


system.part.clear() # remove all particles
p1 = system.part.add(pos=[0, 0, 1], q=+1.0)
p2 = system.part.add(pos=[3, 5, 6], q=-1.0)

pos_vec = p2.pos - p1.pos
analytic_force = -p1.q * p2.q / np.linalg.norm(pos_vec)**3 * pos_vec
print(analytic_force)

# Calculate forces
elc_force = get_elc_forces(p3m, gap_size, pw_error, system)
print(elc_force)
p3m_force = get_p3m_forces(p3m1, gap_size, pw_error, system)
print(p3m_force)

print("FINAL COMPARISON")
print(f"{analytic_force = }")
print(f"{elc_force = }")
print(f"{p3m_force = }")
