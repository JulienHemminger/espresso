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
from elc.src.get_legacy_elc import get_legacy_elc_energy, get_legacy_elc_forces

l_xy = 100.0 # keep l_xy <= 200
l_z = 12.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Parameters for both methods + Initialize P3M deterministically
pw_error = 1e-6
gap_size = 2.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)


# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position, distance r
import matplotlib.pyplot as plt
import numpy as np
import math
from common.generate_constrained_position_pairs import get_rdm_constrained_point_pairs
from elc.src.get_elc_energy import get_elc_energy
from elc.src.get_elc_forces import get_elc_forces



test_count = 3

# Lists to store data for plotting
r_values = []
legacy_energies = []
elc_energies = []

for pos1, pos2 in get_rdm_constrained_point_pairs(test_count):
    R = np.array(pos1) - np.array(pos2)
    r = math.dist(pos1, pos2)
    assert r >= 1
    
    system.part.clear() # remove all particles
    system.part.add(pos=pos1, q=+1.0)
    system.part.add(pos=pos2, q=-1.0)

    # Calculate forces
    ana_force = (-1.0 / r**3) * R
    
    legacy_forces = get_legacy_elc_forces(p3m, gap_size, pw_error, system)
    legacy_force = legacy_forces[np.argmax(np.dot(legacy_forces, ana_force) / np.linalg.norm(legacy_forces, axis=1))]

    elc_forces = get_elc_forces(p3m, gap_size, pw_error, system)
    elc_force = elc_forces[np.argmax(np.dot(elc_forces, ana_force) / np.linalg.norm(elc_forces, axis=1))]

    # Append to lists
    r_values.append(r)

    legacy_error = np.linalg.norm(legacy_force - ana_force)
    legacy_energies.append(legacy_error)
    
    elc_error = np.linalg.norm(elc_force - ana_force)
    elc_energies.append(elc_error)

# Convert to numpy arrays and sort by r to ensure the lines are drawn correctly
sort_idx = np.argsort(r_values)
r_values = np.array(r_values)[sort_idx]
legacy_energies = np.array(legacy_energies)[sort_idx]
elc_energies = np.array(elc_energies)[sort_idx]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r_values, legacy_energies, label='Legacy Force Error', linestyle=':', marker='o', markersize=4)
plt.plot(r_values, elc_energies, label='ELC Force Error', linestyle=':', marker='x', markersize=4)

plt.xlabel(r'Distance $r$')
plt.ylabel(r'Force Error')
plt.title('Comparison of ELC Force Methods vs. Distance')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Save and show
impl_version = get_elc_energy.__module__.split('.')[-1]
print("Finished evaluating "+impl_version)
plt.savefig(f'iX_test1_force_n{test_count}_plot.png')
plt.show()