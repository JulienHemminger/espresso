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

import espressomd
import espressomd.electrostatics
from elc.src.get_elc_energy import get_elc_energy
from elc.src.get_legacy_elc import get_legacy_elc_energy

test_count=20
l_xy = 100.0
l_z = 10.0
gap_size = 1.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

pw_error = 1e-6
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, check_neutrality=False)

# Storage
r_sum_values = [] # Using sum of distances as a proxy for the x-axis
legacy_errors = []
elc_errors = []


charges = [1.0, -1.0, -1.0]

for _ in range(test_count):
    # Generating random positions within the constrained volume
    limit = l_z - gap_size - 0.1
    pos = [np.random.uniform(0, limit, 3) for _ in range(3)]
    
    system.part.clear()
    for i in range(3):
        system.part.add(pos=pos[i], q=charges[i])

    # --- Analytical Energy Calculation (Coulomb Sum) ---
    # E = (q1*q2)/r12 + (q1*q3)/r13 + (q2*q3)/r23
    r12 = math.dist(pos[0], pos[1])
    r13 = math.dist(pos[0], pos[2])
    r23 = math.dist(pos[1], pos[2])
    
    # Avoid singularities
    if any(r < 0.5 for r in [r12, r13, r23]): continue

    ana_energy = (charges[0]*charges[1])/r12 + \
                    (charges[0]*charges[2])/r13 + \
                    (charges[1]*charges[2])/r23

    # --- ESPResSo Calculations ---
    legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
    elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

    # Store Results
    r_sum_values.append(r12 + r13 + r23)
    elc_errors.append(elc_energy - ana_energy)
    legacy_errors.append(legacy_energy - ana_energy)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))

ax.scatter(r_sum_values, elc_errors, color='#2980b9', s=30, label='ELC', alpha=0.6)
ax.scatter(r_sum_values, legacy_errors, color="#ff0000", s=30, marker='s', label='Legacy', alpha=0.6)

ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel(r'Diff ($\Delta E$)')
ax.set_xlabel('Sum of pairwise distances')
ax.set_title('Residuals: Triplet System (+1, -1, -1)', fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
