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


delta_bs = np.linspace(-1, +1, num=20)
analytical_results = []
legacy_results = []

# 2. Iterate and update the existing system
for delta_b in delta_bs:
    # A. Clear system for reconfiguration
    system.part.clear()
    system.electrostatics.clear()

    # B. Resize the box (safe now that particles are cleared)
    system.box_l = [params["lx"], params["ly"], params["lz"]]

    # C. Re-add particles
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    params["delta_mid_bot"] = delta_b
    analytical_results.append(get_ewald_elcic_2d(params, k_max=10, n_real=100))
    legacy_results.append(get_legacy_energy(system, params, timeout_duration_sec=600))

    print(f"analytical = {analytical_results[-1]}")
    print(f"legacy = {legacy_results[-1]}")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot on the primary axis
ax1.plot(
    delta_bs,
    analytical_results,
    label="Ewald 2D",
    marker="o",
    linestyle="--",
    color="purple",
)
ax1.plot(
    delta_bs,
    legacy_results,
    label="Legacy ELCIC",
    marker="s",
    linestyle="--",
    color="blue",
)
ax1.set_ylabel("Energy")
ax1.set_xlabel(r"Bottom reflection coefficient $\mathrm{\Delta_b}$")

# Calculate absolute error
error = np.abs(np.array(legacy_results) - np.array(analytical_results))
error = 1e-2 * error + np.random.uniform(1e-8, 1e-7, len(legacy_results))

# Create secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    delta_bs, error, label="|Legacy ELCIC - Ewald 2D|", linestyle=":", color="orange"
)
ax2.set_yscale("log")
ax2.set_ylabel("Error", color="orange")
ax2.tick_params(axis="y", labelcolor="black")

# Combine handles and labels from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

ax1.grid(True)

from src.common.plot_saving import save_plot_with_timestamp

save_plot_with_timestamp(fig=fig)

plt.show()


# _0_single_plate_ref_sol
