# show reference sol: direct sum not accurate anymore, ewald2d matches legacy elc

import espressomd
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d
from src.elc.energy.analytical.large_box_direct_sum import get_direct_sum_energy
from src.elc.energy.legacy_elc_energy import get_legacy_energy

# 1. Initialize the system ONCE
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Define your parameters
params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}


l_xy_values = np.linspace(10, 30, num=20)  # works for max=100, timeout_duration_sec=600
direct_sum_results = []
ewald_results = []
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

    direct_sum_results.append(get_direct_sum_energy(system))
    ewald_results.append(get_ewald_energy_2d(system))
    legacy_results.append(get_legacy_energy(system, params))

    print(f"direct_sum_results = {direct_sum_results[-1]}")
    print(f"legacy_results = {legacy_results[-1]}")

# 3. Plotting
plt.figure(figsize=(10, 6))  # Increased width to accommodate the text


plt.plot(
    l_xy_values,
    direct_sum_results,
    label="Direct Sum without PBC",
    marker="o",
    linestyle="--",
    color="purple",
)
plt.plot(
    l_xy_values,
    ewald_results,
    label="Ewald 2D",
    marker="s",
    linestyle="--",
    color="orchid",
)
plt.plot(
    l_xy_values,
    legacy_results,
    label="Legacy",
    marker="x",
    linestyle=(0, (5, 10)),
    color="orange",
)


plt.xlabel(r"$L_{xy}$")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)

plt.show()
