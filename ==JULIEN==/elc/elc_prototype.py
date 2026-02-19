import sys
import os
# ESPResSo build path
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))
# Add the project root (where the 'elc' folder lives)
# Based on your ls output, this is: /home/main/Documents/Career/1_Studium/espresso/==JULIEN==
project_root = "/home/main/Documents/Career/1_Studium/espresso/==JULIEN=="
sys.path.insert(0, project_root)


# %%
import espressomd
import espressomd.electrostatics
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position, distance r

from elc.tests._1_dipole.dipole_rdm_pos_energy_test import dipole_rdm_pos_energy_test
dipole_rdm_pos_energy_test(test_count=8)

# %%
# TEST 2: Compare to analytical 2D Madelung energy of a crystal
#from elc.tests._2_madelung.madelung_energy_test import madelung_energy_test
#madelung_energy_test()


# %%
# TEST 3: Compare with the existing implementation of ELC (elc.cpp) for a wide range of systems, where there are no analytical solutions. Compare energy + all forces.
# TODO do legacy & new use same parameters? Sieht bisschen so aus, als würden die Korrekturen unterschiedlich genau berechnet werden
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from common.generate_constrained_position_pairs import generate_constrained_pairs
from elc.impl.get_elc_energy import get_elc_energy

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.impl.get_legacy_elc import get_legacy_elc_energy

# dipole variants (varying gap_size, l_x, l_y, l_z)
import numpy as np
import matplotlib.pyplot as plt
import espressomd
import espressomd.electrostatics
import math

# --- Setup and Helper Functions ---

def run_comparison(l_xy, l_z, gap, p3m_params, system):
    system.part.clear()
    system.box_l = [l_xy, l_xy, l_z]
    
    # Place a simple dipole in the center of the allowed region
    z_mid = (l_z - gap) / 2.0
    system.part.add(pos=[l_xy/2, l_xy/2, z_mid - 0.5], q=+1.0)
    system.part.add(pos=[l_xy/2, l_xy/2, z_mid + 0.5], q=-1.0)

    # Initialize P3M (ensure it is tuned/updated for new box)
    p3m = espressomd.electrostatics.P3M(**p3m_params)
    
    # Calculate energies
    legacy_e = get_legacy_elc_energy(p3m, gap, p3m_params['accuracy'], system)
    
    # Newer ELC interface (post ESPResSo 4.2 style)
    newer_e = get_elc_energy(p3m, gap, p3m_params['accuracy'], system)
    
    return legacy_e, newer_e

# --- Variation Loops ---
test_count_per_variable = 3
params = {'accuracy': 1e-6, 'prefactor': 1.0, 'epsilon': 1.0}
system = espressomd.System(box_l=[100, 100, 10])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Define ranges to test
test_ranges = {
    "gap_size": np.linspace(1.0, 5.0, test_count_per_variable),
    "l_xy": np.linspace(50.0, 200.0, test_count_per_variable),
    "l_z": np.linspace(8.0, 20.0, test_count_per_variable)
}

results = {k: {"legacy": [], "newer": []} for k in test_ranges}

# 1. Vary Gap Size
for g in test_ranges["gap_size"]:
    leg, new = run_comparison(100.0, 10.0, g, params, system)
    results["gap_size"]["legacy"].append(leg)
    results["gap_size"]["newer"].append(new)

# 2. Vary L_xy (keeping L_z and gap constant)
for l in test_ranges["l_xy"]:
    leg, new = run_comparison(l, 10.0, 2.0, params, system)
    results["l_xy"]["legacy"].append(leg)
    results["l_xy"]["newer"].append(new)

# 3. Vary L_z
for lz in test_ranges["l_z"]:
    leg, new = run_comparison(100.0, lz, 2.0, params, system)
    results["l_z"]["legacy"].append(leg)
    results["l_z"]["newer"].append(new)

# --- Plotting ---
titles = ["Varying Gap Size", "Varying $L_{xy}$", "Varying $L_z$"]
x_labels = ["Gap Size", "$L_{xy}$", "$L_z$"]
keys = ["gap_size", "l_xy", "l_z"]

# --- Modified Plotting Script ---
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
for i, key in enumerate(keys):
    x = test_ranges[key]
    diff = np.array(results[key]["newer"]) - np.array(results[key]["legacy"])
    axs[i].plot(x, diff, color='#e2402e', marker='o', linestyle='-', linewidth=1.5)
    axs[i].axhline(0, color='black', lw=1, ls='--')
    axs[i].set_title(f"Residuals: {titles[i]}")
    axs[i].set_ylabel(r"$\Delta E$ (Newer - Legacy)")
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""



"""

* madelung variants (box_size, n_ions/spacing, etc.)

* ?
    * completely randomized systems? (random box_sizes, gap size, particle count, -positions, )
    * non-neutral systems?
"""
