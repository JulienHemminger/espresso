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
pw_error = 1e-6
gap_size = 2.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)

# %%
def get_elc_energy(p3m, gap_size, pw_error, system):
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    return e_3d

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole.
l_xy = 100.0 # keep l_xy <= 200
l_z = 10.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

r = 5.0 # 1 - 10
system.part.add(pos=[0.0, 0.0, 0.0], q=+1.0)
system.part.add(pos=[0.0, 0.0, r], q=-1.0)

# Computation - TODO force
ana_energy = -1.0/r

legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

print(f"r = {r}")
print(f"* {legacy_energy=}")
print(f"* {elc_energy=}")
print(f"* {ana_energy=}")


# %%
# TEST 2: Compare to analytical solution(energy, force) for a dipole at different box sizes.

# %%
# TEST 3: Compare to analytical solution(energy, force) for a dipole at different z-values.


# %%
# TEST 4: Compare to analytical 2D Madelung energy of a crystal (i think forces cant be calculated analytically anymore).
# Alex: Die Madelungen Energie ist halt die Energie pro Teilchen in einem unendlichen Kristall. Die konvergiert zu einer Konstanten, der Madelungen-Konstanten. Kann man analytisch zeigen, gibt in Espresso auch ein Testcase dazu. Kannst auch mal reinschauen

# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)
