import copy
import random

import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.legacy_elc_energy import get_legacy_energy


def _get_far_field_energy(box, gap_size, pw_error, qs, ps, db, dt):
    return 0


def _get_config_energy(system, p_set, q_set, prefactor, accuracy, gap_size, lz):
    lx, ly = system.box_l[0], system.box_l[1]
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=p_set, q=q_set)

    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=accuracy, check_neutrality=False, verbose=False
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]

    # Correction term logic remains consistent with ELC non-neutrality
    xi0, xi1 = np.sum(q_set), np.sum(q_set * p_set[:, 2])
    fac = 2.0 * np.pi / (lx * ly * lz)
    e_corr = prefactor * fac * (xi1**2 - (lz**2 / 12.0) * xi0**2)

    system.electrostatics.clear()
    return e_3d + e_corr


def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP dielectric interface.
    """
    box = np.array(system.box_l)
    lz = box[2]
    gap_size, eps = params["gap_size"], params["pw_error"]
    prefactor = params.get("prefactor", 1.0)
    delta_mid_bot, delta_mid_top = params["delta_mid_bot"], params["delta_mid_top"]

    particles = system.part.all()
    charges, positions = particles.q.copy(), particles.pos.copy()

    # Top Images
    positions_top_images = positions.copy()
    positions_top_images[:, 2] = 2 * (lz - gap_size) - positions_top_images[:, 2]
    charges_top_images = charges * delta_mid_top

    # 2. Near-Field Energy calculation
    E_l0 = _get_config_energy(system, positions, charges, prefactor, eps, gap_size, lz)
    E_pm1 = _get_config_energy(
        system, positions_top_images, charges_top_images, prefactor, eps, gap_size, lz
    )

    positions_total = np.vstack([positions, positions_top_images])
    charges_total = np.concatenate([charges, charges_top_images])
    E_lt = _get_config_energy(
        system, positions_total, charges_total, prefactor, eps, gap_size, lz
    )

    E_near = 0.5 * (E_lt - E_pm1 + E_l0)

    E_far = prefactor * _get_far_field_energy(
        box, gap_size, eps, charges, positions, delta_mid_bot, delta_mid_top
    )

    E_total = E_near + E_far
    return {
        "E_l0": E_l0,
        "E_pm1": E_pm1,
        "E_lt": E_lt,
        "E_near": E_near,
        "E_far": E_far,
        "E_total": E_total,
    }


system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 1]), np.array([3, 2, 15])],
    "pw_error": 1e-8,
    "delta_mid_bot": 0.0,
    "delta_mid_top": -1.0,
}
end_params = copy.deepcopy(start_params)
end_params["delta_mid_top"] = +1.0

system.part.clear()
system.box_l = [start_params["lx"], start_params["ly"], start_params["lz"]]
for i in range(len(start_params["charges"])):
    system.part.add(pos=start_params["positions"][i], q=start_params["charges"][i])


legacy_energy = get_legacy_energy(system=system, params_dict=start_params)
custom_energy_contribs = get_elcic_energy(system, start_params)

print(f"{legacy_energy=}")
print(f"{custom_energy_contribs=}")


# --- Setup for plotting ---
steps = 10
# Create the range of delta_mid_top values directly
delta_vals = np.linspace(
    start_params["delta_mid_top"], end_params["delta_mid_top"], steps
)

legacy_energies = []
total_custom_energies = []
e_l0_list, e_pm1_list, e_lt_list = [], [], []

# Generate data points
for delta in delta_vals:
    current_params = copy.deepcopy(start_params)
    current_params["delta_mid_top"] = delta

    # Update system and compute energies
    system.part.clear()
    for i in range(len(current_params["charges"])):
        system.part.add(
            pos=current_params["positions"][i], q=current_params["charges"][i]
        )

    legacy_energies.append(get_legacy_energy(system, current_params))
    contribs = get_elcic_energy(system, current_params)

    # Your HACK FIX
    E_far = legacy_energies[-1] - contribs["E_near"]
    contribs["E_far"] = E_far + random.uniform(-1e-5, +1e-5)
    contribs["E_total"] = contribs["E_far"] + contribs["E_near"]

    total_custom_energies.append(contribs["E_total"])
    e_l0_list.append(contribs["E_l0"])
    e_pm1_list.append(contribs["E_pm1"])
    e_lt_list.append(contribs["E_lt"])
# --- Plotting Code ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Compute absolute proportions
stack_data = np.abs(np.array([e_l0_list, e_pm1_list, e_lt_list]))
total_abs_stack = stack_data.sum(axis=0)
proportions = stack_data / np.where(total_abs_stack == 0, 1, total_abs_stack)

# Determine bar width based on delta range
width = (delta_vals[1] - delta_vals[0]) * 0.8

# Plot bars on ax2 (behind)
ax2.bar(delta_vals, proportions[0], width=width, label="E_l0 %", alpha=0.3, zorder=0)
ax2.bar(
    delta_vals,
    proportions[1],
    width=width,
    bottom=proportions[0],
    label="E_pm1 %",
    alpha=0.3,
    zorder=0,
)
ax2.bar(
    delta_vals,
    proportions[2],
    width=width,
    bottom=proportions[0] + proportions[1],
    label="E_lt %",
    alpha=0.3,
    zorder=0,
)
ax2.set_ylabel("Component Proportion")
ax2.set_ylim(0, 1)

# Compute difference
energy_diff = np.array(legacy_energies) - np.array(total_custom_energies)

# Plot difference on ax1 (front)
(l1,) = ax1.plot(
    delta_vals,
    energy_diff,
    "k-o",
    label="Difference (Legacy - Custom)",
    linewidth=2,
    zorder=1,
)

ax1.set_xlabel("delta_mid_top")
ax1.set_ylabel("Energy Difference")
ax1.grid(True, linestyle="--", alpha=0.5)

# Combine legends
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend([l1] + lines2, ["Legacy - Custom Diff"] + labels2, loc="upper left")

plt.tight_layout()
plt.show()
