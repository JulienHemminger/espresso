import espressomd
import matplotlib.pyplot as plt
import numpy as np
from src.common.plot_saving import save_plot_with_timestamp
from src.elc.energy.analytical.large_box_direct_sum import (
    get_direct_sum_energy as get_direct_sum_energy,
)
from src.elc.energy.custom_elc_energy import get_elc_energy


def run_madelung(system, ions_list, gap_size=1, accuracy=1e-12):
    madelung_refs = []
    elc_energies = []

    for ions_per_axis in ions_list:
        system.part.clear()
        system.box_l = [10, 10, 10]

        l_xy = min(system.box_l[0], system.box_l[1])

        spacing = l_xy / ions_per_axis
        ion_z_pos = system.box_l[2] / 2.0
        for i in range(ions_per_axis):
            for j in range(ions_per_axis):
                charge = (-1.0) ** (i + j)
                system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

        ion_count = len(system.part)
        MADELUNG_CONSTANT_PERFECT_2D_SHEET = (
            1.6155426267128247  # e.g. perfect 2D sheet of NaCl
        )
        madelung_2d_ref = (
            -MADELUNG_CONSTANT_PERFECT_2D_SHEET * ion_count / (2.0 * spacing)
        )
        madelung_refs.append(madelung_2d_ref)

        elc_energy = get_elc_energy(system, gap_size, accuracy)
        elc_energies.append(elc_energy)

        print(f"{ions_per_axis=}: diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies


"""
Error caused by
* custom elc: NO (i get same error with legacy_elc)
* reference_energy: YES
"""


# Setup
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

ions_range = range(2, 20 + 1, 4)  # Testing different grid sizes
refs, elcs = run_madelung(system, ions_range)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-axis: Energy
color = "dodgerblue"
ax1.set_xlabel("Ions per Axis")
ax1.set_ylabel("Energy", color=color)
ax1.plot(
    ions_range,
    refs,
    label="Reference",
    marker="x",
    linestyle="solid",
    color="purple",
)
ax1.plot(ions_range, elcs, marker="o", color=color, label="Custom", linestyle="dashed")
ax1.tick_params(axis="y", labelcolor=color)


# Secondary Y-axis: Absolute Difference
ax2 = ax1.twinx()
color = "coral"
abs_diff = np.abs(np.array(refs) - np.array(elcs))
ax2.set_ylabel("abs(Difference)", color=color)
ax2.plot(ions_range, abs_diff, "s:", color=color, label="Difference")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_yscale("log")  # Often useful to see differences on a log scale

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

# Combine them and place in a single location
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")

plt.grid(True, which="both", axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

save_plot_with_timestamp(fig=fig)
plt.show()
