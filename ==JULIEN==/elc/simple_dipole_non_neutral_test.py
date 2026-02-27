# %%
import sys
import os
espresso_path = "/home/main/Documents/Career/1_Studium/espresso"
sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))
sys.path.insert(0, os.path.join(espresso_path, "==JULIEN=="))

# %%
import matplotlib.pyplot as plt
import numpy as np
import math
from common.get_positions import get_rdm_constrained_point_pairs
from elc.src.get_elc_energy import get_elc_energy

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy
import random

test_count = 10
l_xy = 100.0 # keep l_xy <= 200
l_z = 10.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Parameters for both methods + Initialize P3M deterministically
pw_error = 1e-6
gap_size = 1.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, check_neutrality=False)

# Lists to store data for plotting
r_values = []
legacy_energies = []
elc_energies = []
ana_energies = []

# Neutrales System:
# ohne Dipolmoment: err=1e-4
# mit Dipolmoment: err=1e-4

# Nicht-Neutrales System:
# ohne Dipolmoment: Fehler=0.08
# mit Dipolmoment: Fehler=0.08


for pos1, pos2 in get_rdm_constrained_point_pairs(test_count, box_size=min(l_xy, l_z-gap_size-1e-3)):
    pos1 = (l_xy/2, l_xy/2, 1.0)
    pos2 = (l_xy/2, l_xy/2, l_z-gap_size-1.0)
    r = math.dist(pos1, pos2)
    assert r >= 1
    
    q1 = +1.0 #random.uniform(-1.0, +1.0)
    q2 = +1.0 #random.uniform(-1.0, +1.0)
    system.part.clear()
    system.part.add(pos=pos1, q=q1)
    system.part.add(pos=pos2, q=q2)

    # Calculate energies
    ana_energy = q1 * q2 / r
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # Append to lists
    r_values.append(r)
    legacy_energies.append(legacy_energy)
    elc_energies.append(elc_energy)
    ana_energies.append(ana_energy)

# Convert to numpy arrays and sort by r to ensure the lines are drawn correctly
sort_idx = np.argsort(r_values)
r_values = np.array(r_values)[sort_idx]
legacy_energies = np.array(legacy_energies)[sort_idx]
elc_energies = np.array(elc_energies)[sort_idx]
ana_energies = np.array(ana_energies)[sort_idx]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 4))

# --- Residual Plot ---
elc_error = np.array(elc_energies) - np.array(ana_energies)
legacy_error = np.array(legacy_energies) - np.array(ana_energies)

# Added labels, distinct markers ('o' and 's'), and transparency (alpha)
ax.scatter(r_values, elc_error, color='#2980b9', s=30, marker='o', alpha=0.6, label='ELC')
ax.scatter(r_values, legacy_error, color="#ff0000", s=30, marker='s', alpha=0.6, label='Legacy')

ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Formatting
ax.set_ylabel(r'Diff ($\Delta E$)')
ax.set_xlabel(r'Inter-particle distance ($r$)')
ax.set_title('Residuals of Energy Computation', fontweight='bold', pad=10)

# Display the legend to show the labels
ax.legend(frameon=False)

# Styling
ax.grid(True, linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()