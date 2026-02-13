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
from elc.get_legacy_elc import get_legacy_elc_energy

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position, distance r
"""
from elc.tests.dipole_rdm_pos_energy_test import dipole_rdm_pos_energy_test
dipole_rdm_pos_energy_test(test_count=3)
"""
# %%
# TEST 2: Compare to analytical solution(energy, force) for a dipole at different gap_size?, l_xy?, l_z



# %%
# TEST 4: Compare to analytical 2D Madelung energy of a crystal (i think forces cant be calculated analytically anymore).
# Alex: Die Madelungen Energie ist halt die Energie pro Teilchen in einem unendlichen Kristall. Die konvergiert zu einer Konstanten, der Madelungen-Konstanten. Kann man analytisch zeigen, gibt in Espresso auch ein Testcase dazu. Kannst auch mal reinschauen

# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)
