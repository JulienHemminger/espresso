import sys

import os

espresso_path = "/home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso"

sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))

sys.path.insert(0, os.path.join(espresso_path, "==JULIEN==")) 

import sys
import os
import espressomd
import espressomd.electrostatics
import numpy as np
import matplotlib.pyplot as plt

# Your specific paths
espresso_path = "/home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso"
sys.path.insert(0, os.path.join(espresso_path, "build", "src", "python"))
sys.path.insert(0, os.path.join(espresso_path, "==JULIEN=="))

from elc.src.third_party.get_ewald_energy_2d import direct_sum_energy

def get_energies(n_values):
    # 1. System Setup
    l_xy = 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_xy, l_xy, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    # Clear existing particles and add two opposite charges
    system.part.clear()
    system.part.add(pos=[5.0, 5.0, 1.0], q=1.0)
    system.part.add(pos=[5.0, 5.0, 2.0], q=-1.0) # Changed to -1.0 for a typical dipole interaction

    direct_energies = []

    # 2. Calculation Loop
    for n in n_values:
        n_int = int(n)
        print(f"Calculating Direct Sum for n_max = {n_int}")
        
        # Calculate and append
        e_direct = direct_sum_energy(system, n_int)
        direct_energies.append(e_direct)

    return direct_energies

# 3. Execution and Plotting
n_range = np.geomspace(10, 1000, num=10)
direct_res = get_energies(n_range)

plt.figure(figsize=(10, 6))
plt.plot(n_range, direct_res, 's--', color='orange', label='Direct Sum Energy')
plt.xscale('log') # Added log scale since n_range is geometric
plt.xlabel('$n_{max}$ (Image shells)')
plt.ylabel('Interaction Energy')
plt.title('Direct Sum Convergence in 2D Slab Geometry')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()