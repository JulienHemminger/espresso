# %%
import sys
import os
espresso_path = "/home/main/Documents/Career/1_Studium/espresso"
sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))
sys.path.insert(0, os.path.join(espresso_path, "==JULIEN=="))

# %%
import numpy as np
from common.get_positions import get_rdm_constrained_points
from elc.src.get_elc_energy import get_elc_energy

import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.src.get_legacy_elc import get_legacy_elc_energy
import numpy as np
import espressomd


import numpy as np
import espressomd
from itertools import combinations

l_x = 100.0 # keep l_xy <= 200
l_y = 100.0
l_z = 10.0

system = espressomd.System(box_l=[l_x, l_y, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Parameters for both methods + Initialize P3M deterministically
pw_error = 1e-6
gap_size = 1.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error, check_neutrality=False)


point_count = 3

rs = get_rdm_constrained_points(l_x, l_y, l_z-gap_size-1e-3, point_count)
qs = [+1.0, -1.0, -1.0]
for i in range(min(len(rs), len(qs))):
    system.part.add(pos=rs[i], q=qs[i])

# --- Analytical Energy Calculation ---
particles = list(system.part.all())
analytical_energy = sum((p1.q * p2.q) / np.linalg.norm(p1.pos - p2.pos) for p1, p2 in combinations(particles, 2))


# Calculate energies
legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))

# --- Comparison and Output ---
print(f"\nResults for {point_count} particles:")
print(f"{'Method':<20} | {'Energy':<15}")
print("-" * 38)
print(f"{'Analytical':<20} | {analytical_energy:<15.6f}")
print(f"{'Legacy ELC':<20} | {legacy_energy:<15.6f}")
print(f"{'New ELC':<20} | {elc_energy:<15.6f}")