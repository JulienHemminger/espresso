import espressomd
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.analytical.large_box_direct_sum import (
    get_direct_sum_energy as get_direct_sum_energy,
)
from src.elc.energy.legacy_elc_energy import get_legacy_energy

# 1. Initialize the system ONCE
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Define your parameters
params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 20.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}


l_xy_values = np.linspace(20, 60, num=20)  # works for max=100, timeout_duration_sec=600
analytical_results = []
legacy_results = []

# 2. Iterate and update the existing system
for l_xy in l_xy_values:
    # A. Clear system for reconfiguration
    system.part.clear()
    system.electrostatics.clear()

    # B. Resize the box (safe now that particles are cleared)
    system.box_l = [l_xy, l_xy, params["lz"]]

    # C. Re-add particles
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    # D. Calculate energies
    params["lx"] = l_xy
    params["ly"] = l_xy

    analytical_results.append(get_direct_sum_energy(system))
    legacy_results.append(get_legacy_energy(system, params, timeout_duration_sec=600))

    print(f"analytical = {analytical_results[-1]}")
    print(f"legacy = {legacy_results[-1]}")

# 3. Plotting
plt.figure(figsize=(10, 6))  # Increased width to accommodate the text


plt.plot(
    l_xy_values,
    analytical_results,
    label="Analytical",
    marker="o",
    linestyle="--",
    color="purple",
)
plt.plot(
    l_xy_values,
    legacy_results,
    label="Legacy ELC",
    marker="s",
    linestyle="--",
    color="orange",
)

plt.xlabel(r"$L_{xy}$")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)

plt.show()
