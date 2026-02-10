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


# Parameters for both methods + Initialize P3M deterministically
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=1e-3, mesh=[32, 32, 32], cao=3, alpha=0.35, r_cut=4.5)
gap_size = 2.0
pw_error = 1e-3

# %%
def get_p3m_energy(p3m, gap_size, pw_error, system):
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    
    return e_3d

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole.
l_xy = 1_000.0
l_z = 10.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

r = 4.0
system.part.add(pos=[0.0, 0.0, 0.0], q=+1.0)
system.part.add(pos=[0.0, 0.0, r], q=-1.0)

# Computation - TODO force
ana_energy = -1.0/r # $$U = \frac{1}{4\pi\varepsilon_0} \frac{q_1 q_2}{r}$$

elc_energy = get_p3m_energy(p3m, gap_size, pw_error, system)

print(get_legacy_elc_energy(p3m, gap_size, pw_error, system))
print(elc_energy)

# %%
# TEST 2: Compare to analytical solution(energy, force) for a dipole at different box sizes.

# %%
# TEST 3: Compare to analytical solution(energy, force) for a dipole at different z-values.


# %%
# TEST 4: Compare to analytical 2D Madelung energy of a crystal (i think forces cant be calculated analytically anymore).

# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)
"""
system.part.add(pos=[1.0, 5.0, 1.0], q=+1.0)
system.part.add(pos=[7.0, 5.0, 1.0], q=-1.0)
system.analysis.energy() # "step 0" to update forces

elc_energy = get_elc_energy(p3m, gap_size, pw_error, system)
elc_force = get_elc_force(p3m, gap_size, pw_error, system)


legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
legacy_force = get_legacy_elc_force(p3m, gap_size, pw_error, system)


assert math.isclose(legacy_energy, elc_energy, abs_tol=1e-2), f"{legacy_energy=} != {elc_energy=}"
"""
