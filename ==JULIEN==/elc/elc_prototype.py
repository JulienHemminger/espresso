import sys
import os
espresso_path = "/home/main/Documents/Career/1_Studium/espresso"
sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))
sys.path.insert(0, os.path.join(espresso_path, "==JULIEN=="))

import espressomd
import espressomd.electrostatics
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position
#from elc.tests._1_dipole.dipole_rdm_pos_energy_test import dipole_rdm_pos_energy_test
#dipole_rdm_pos_energy_test(test_count=4*8)

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
NO = []
f"""

* improve my elc
    * maybe give gemini elc.cpp as info 

    * maybe "decompose" it further: {NO} more tests, more problems/code/parts/errors
        * TEST4: "q_i is not always +-1"
        * TEST5: "random particle count (neutral system)"
        * TEST6: "random particle count (non-neutral system)
* energy contributions plot

"""
import matplotlib.pyplot as plt
import numpy as np
import espressomd
import espressomd.electrostatics
from elc.src.get_elc_energy import get_elc_energy
from elc.src.get_legacy_elc import get_legacy_elc_energy

# --- Setup ---
p3m_params = {'accuracy': 1e-6, 'prefactor': 1.0, 'epsilon': 1.0, 'check_neutrality': False}
system = espressomd.System(box_l=[50.0, 50.0, 20.0])
system.time_step = 0.01
system.cell_system.skin = 0.4

def run_comparison_with_random_particles(n_particles, l_xyz, gap, p3m_params):
    system.part.clear()
    system.box_l = [l_xyz, l_xyz, l_xyz]
    
    # Randomly place particles with random charges
    # Charges range from -2.0 to 2.0 to allow for net variations
    for _ in range(n_particles):
        pos = np.random.rand(3) * l_xyz

        pos[2] =np.random.rand() * (l_xyz - gap - 1e-3)
        q = np.random.uniform(-2.0, 2.0)
        system.part.add(pos=pos, q=q)

    p3m = espressomd.electrostatics.P3M(**p3m_params)
    legacy_e = get_legacy_elc_energy(p3m, gap, p3m_params['accuracy'], system)
    newer_e = get_elc_energy(p3m, gap, p3m_params['accuracy'], system)
    
    return legacy_e, newer_e

# --- Execution ---
particle_counts = range(3, 8+1)
results = {"legacy": [], "newer": []}

for n in particle_counts:
    leg, new = run_comparison_with_random_particles(n, 50.0, 2.0, p3m_params)
    results["legacy"].append(leg)
    results["newer"].append(new)

# --- Plotting ---
diff = np.array(results["newer"]) - np.array(results["legacy"])

plt.figure(figsize=(8, 5))
plt.plot(particle_counts, diff, color='#e2402e', marker='o', linestyle='-')
plt.axhline(0, color='black', lw=1, ls='--')
plt.title("Residuals vs. Particle Count (Variable Charges)")
plt.xlabel("Number of Particles")
plt.ylabel(r"$\Delta E$ (Newer - Legacy)")
plt.grid(True, alpha=0.3)
plt.show()