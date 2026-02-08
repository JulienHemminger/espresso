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

# Initialize an empty system
box_l = 10.0
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Initialize P3M deterministically
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=1e-3, mesh=[32, 32, 32], cao=3, alpha=0.35, r_cut=4.5)

# Parameters for both methods
gap_size = 2.0
pw_error = 1e-3

def get_elc_energy():
    """Calculates analytical Coulomb energy between particle 0 and 1."""
    p1 = system.part.by_id(0)
    p2 = system.part.by_id(1)
    
    # Distance calculation using system properties
    dist = np.linalg.norm(p1.pos - p2.pos)
    
    # U = prefactor * q1 * q2 / r
    energy = (1.0 * p1.q * p2.q) / dist
    return energy

def get_elc_force():
    """Calculates analytical Coulomb force on particle 0 from particle 1."""
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
# TEST 1: Analytical calculation of energy and force for a dipole
system.part.add(pos=[5.0, 5.0, 1.0], q=+1.0)
system.part.add(pos=[5.0, 5.0, 7.0], q=-1.0)

analytical_energy = get_elc_energy()
analytical_force_p1 = get_elc_force()

print(f"Analytical Energy: {analytical_energy}")
print(f"Analytical Force on P1: {analytical_force_p1}")
"""
--- Analytical Results (Dipole) ---
Energy:   -0.1667
Force P1: [-0.0, -0.0, 0.027777777777777776]
"""


# %%
# TEST 2: Force and energy of a dipole at different box sizes.

# %%
# TEST 3: Force and energy of a dipole at different z-values.


# %%
# TEST 4: Calculate the 2D Madelung energy of a crystal.
# %%
# TEST 5: Compare with the existing implementation of ELC.
"""
def get_legacy_ELC_energy(actor, gap_size, pw_error):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error
    )
    system.electrostatics.solver = elc_legacy
    return system.analysis.energy()['total']

legacy_elc = get_legacy_ELC_energy(p3m, gap_size, pw_error) # -0.023799
newer_elc = get_newer_ELC_energy(p3m, gap_size, pw_error)
assert math.isclose(legacy_elc, newer_elc, abs_tol=1e-3), f"{legacy_elc=} != {newer_elc=}"
"""
