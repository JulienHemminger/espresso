# %%
import sys
import os
espresso_path = "/home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso"
sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))
sys.path.insert(0, os.path.join(espresso_path, "==JULIEN=="))

import espressomd  # pyright: ignore[reportMissingImports]
import espressomd.electrostatics  # pyright: ignore[reportMissingImports]
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position
#from elc.tests._1_dipole.dipole_rdm_pos_energy_test import dipole_rdm_pos_energy_test
#dipole_rdm_pos_energy_test(test_count=1)

#from elc.tests._1_dipole.dipole_rdm_pos_energy_non_square_test import dipole_rdm_pos_energy_non_square_test
#dipole_rdm_pos_energy_non_square_test()

# %%
# TEST 2: Compare to analytical 2D Madelung energy of a crystal
#from elc.tests._2_madelung.madelung_energy_test import madelung_energy_test
#madelung_energy_test()

# %%
# TEST 3: Compare with the existing implementation of ELC (elc.cpp) for a wide range of systems, where there are no analytical solutions. Compare energy + all forces.
#from elc.tests._3_general_systems.dipole_variants_test import dipole_variants_test
#dipole_variants_test()
# %%
# TEST 4: particle_count=3-10, non-neutral systems, varying charges q_i
from elc.tests._3_general_systems.non_neutral_system_test import run_test
run_test()
