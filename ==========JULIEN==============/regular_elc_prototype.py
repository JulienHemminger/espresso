# %%
import sys
import os
# To allow me to write/run juptyer notebooks from VScode.
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))

# %%
import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.legacy_elc import get_legacy_elc_energy, get_legacy_elc_force

# Initialize an empty system
box_l = 10.0
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Parameters for both methods
gap_size = 2.0
pw_error = 1e-3

# %%
def get_elc_energy():
    p1 = system.part.by_id(0)
    p2 = system.part.by_id(1)
    
    # Distance calculation using system properties
    dist = np.linalg.norm(p1.pos - p2.pos)
    
    # U = prefactor * q1 * q2 / r
    energy = (1.0 * p1.q * p2.q) / dist
    return energy

def get_elc_force():
    p1 = system.part.by_id(0)
    p2 = system.part.by_id(1)
    
    # Vector from p2 to p1
    vec_r12 = p1.pos - p2.pos
    dist = np.linalg.norm(vec_r12)
    
    # Unit vector
    unit_vec = vec_r12 / dist
    
    # F = (prefactor * q1 * q2 / r^2) * unit_vector
    force_mag = (1.0 * p1.q * p2.q) / (dist**2)
    force_p1 = force_mag * unit_vec
    
    return force_p1

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole.
system.part.add(pos=[5.0, 5.0, 1.0], q=+1.0)
system.part.add(pos=[5.0, 5.0, 7.0], q=-1.0)
system.analysis.energy() # "step 0" to update forces

elc_energy = get_elc_energy()
elc_force = get_elc_force()


p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=1e-3, mesh=[32, 32, 32], cao=3, alpha=0.35, r_cut=4.5) # Initialize P3M deterministically
legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
legacy_force = get_legacy_elc_force(p3m, gap_size, pw_error, system)

ana_energy = -0.166667
ana_force = np.array([-0.0, -0.0, 0.027776])

assert math.isclose(elc_energy, ana_energy, abs_tol=1e-3)
assert np.allclose(elc_force, ana_force, atol=1e-3)


# TODO: double check analytical solution
# TODO: both assertions below fail. whats wrong? is the legacy elc wrong(or the way im using it)? or analytical solution wrong? maybe even both?
assert math.isclose(legacy_energy, ana_energy, abs_tol=1e-3), f"{legacy_energy=} != {ana_energy=}"
assert np.allclose(legacy_force, ana_force, atol=1e-3), f"{legacy_force=} != {ana_force=}"

# %%
# TEST 2: Compare to analytical solution(energy, force) for a dipole at different box sizes.

# %%
# TEST 3: Compare to analytical solution(energy, force) for a dipole at different z-values.


# %%
# TEST 4: Compare to analytical 2D Madelung energy of a crystal (i think forces cant be calculated analytically anymore).
# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)
