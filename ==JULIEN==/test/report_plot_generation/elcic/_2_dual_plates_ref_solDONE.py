import espressomd
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elcic.energy.single_plate.neutral.metallic.analytical import get_ewald_elcic_2d

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
    "delta_mid_bot": -1.0,
    "delta_mid_top": 0.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}


t_values = np.linspace(-1, 1, num=20)
analytical_results = []
legacy_results = []

# 2. Iterate and update the existing system
for t in t_values:
    # A. Clear system for reconfiguration
    system.part.clear()
    system.electrostatics.clear()

    # B. Resize the box (safe now that particles are cleared)
    system.box_l = [params["lx"], params["ly"], params["lz"]]

    # C. Re-add particles
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    params["delta_mid_bot"] = +t
    params["delta_mid_top"] = -t

    analytical_results.append(get_ewald_elcic_2d(params, k_max=10, n_real=100))
    legacy_results.append(get_legacy_energy(system, params, timeout_duration_sec=600))

    print(f"analytical = {analytical_results[-1]}")
    print(f"legacy = {legacy_results[-1]}")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot on the primary axis
ax1.plot(
    t_values,
    analytical_results,
    label="Ewald 2D",
    marker="o",
    linestyle="--",
    color="purple",
)
ax1.plot(
    t_values,
    legacy_results,
    label="Legacy ELCIC",
    marker="s",
    linestyle="--",
    color="blue",
)
ax1.set_ylabel("Energy")
ax1.set_xlabel(r"Reflection coefficient parameter t")

# Calculate absolute error
error = np.abs(np.array(legacy_results) - np.array(analytical_results))

# Create secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    t_values,
    error,
    label="|Legacy ELCIC - Ewald 2D|",
    linestyle=":",
    color="orange",
)
ax2.set_yscale("log")
ax2.set_ylabel("Error", color="orange")
ax2.tick_params(axis="y")

# Combine handles and labels from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

ax1.grid(True)
from src.common.plot_saving import save_plot_with_timestamp

save_plot_with_timestamp(fig)


plt.show()
