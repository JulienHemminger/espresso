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

l_xy = 10.0 # keep l_xy <= 200
l_z = 10.0

system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Parameters for both methods + Initialize P3M deterministically
pw_error = 1e-6
gap_size = 1.0
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)


# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position, distance r
import matplotlib.pyplot as plt
import numpy as np
import math
from common.generate_constrained_position_pairs import generate_constrained_pairs
from elc.get_elc_energy import get_elc_energy



# Increase test_count for a more descriptive plot
test_count = 10

# Lists to store data for plotting
r_values = []
legacy_energies = []
elc_energies = []
ana_energies = []

for pos1, pos2 in generate_constrained_pairs(test_count, box_size=min(l_xy, l_z-gap_size-1e-3)):
    r = math.dist(pos1, pos2)
    assert r >= 1
    
    system.part.clear() # remove all particles
    system.part.add(pos=pos1, q=+1.0)
    system.part.add(pos=pos2, q=-1.0)

    # Calculate energies
    ana_energy = -1.0 / r
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

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r_values, ana_energies, label='Analytical Energy ($-1/r$)', linestyle=':', color='black', linewidth=2)
plt.plot(r_values, legacy_energies, label='Legacy ELC Energy', linestyle=':', marker='o', markersize=4)
plt.plot(r_values, elc_energies, label='ELC Energy', linestyle=':', marker='x', markersize=4)

plt.xlabel(r'Distance $r$')
plt.ylabel(r'Energy $E$')
plt.title('Comparison of Energy Methods vs. Distance')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Save and show
impl_version = get_elc_energy.__module__.split('.')[-1]
print("Finished evaluating "+impl_version)
plt.savefig(f'{impl_version}_test1_energy_n{test_count}_plot.png')
plt.show()

# %%
# TEST 2: Compare to analytical solution(energy, force) for a dipole at different gap_size?, l_xy?, l_z



# %%
# TEST 4: Compare to analytical 2D Madelung energy of a crystal (i think forces cant be calculated analytically anymore).
# Alex: Die Madelungen Energie ist halt die Energie pro Teilchen in einem unendlichen Kristall. Die konvergiert zu einer Konstanten, der Madelungen-Konstanten. Kann man analytisch zeigen, gibt in Espresso auch ein Testcase dazu. Kannst auch mal reinschauen

# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)
