import espressomd
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.analytical.large_box_direct_sum import (
    get_direct_sum_energy as get_direct_sum_energy,
)
from src.elc.energy.custom_elc_energy import get_elc_energy


def run_madelung(system, box_widths, gap_size=1, accuracy=1e-12):
    madelung_refs = []
    elc_energies = []
    ions_per_axis = 2

    for box_width in box_widths:
        system.part.clear()
        system.box_l = [box_width, box_width, 10.0]  # Scale the box

        # In a fully periodic box, spacing is exactly box_width / ions_per_axis
        spacing = box_width / ions_per_axis
        ion_z_pos = system.box_l[2] / 2.0

        for i in range(ions_per_axis):
            for j in range(ions_per_axis):
                charge = (-1.0) ** (i + j)
                system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

        ion_count = len(system.part)
        MADELUNG_CONSTANT_PERFECT_2D_SHEET = 1.6155426267128247

        madelung_2d_ref = (
            -MADELUNG_CONSTANT_PERFECT_2D_SHEET * ion_count / (2.0 * spacing)
        )
        elc_energy = get_elc_energy(system, gap_size, accuracy)
        elc_energies.append(elc_energy)
        madelung_refs.append(madelung_2d_ref)

        print(f"{box_width=}: diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies


# Setup
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

box_sizes = range(10, 30 + 1, 1)
refs, elcs = run_madelung(system, box_sizes)


from src.common.plot_saving import save_plot_with_timestamp

# Assuming variables box_sizes, refs, and elcs are defined in your context
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-axis: Energy
# Mapping "Reference" to "Analytical / Direct Sum / Ewald 2D" (purple)
# Mapping "ELC" to "Legacy ELC" (orange) as per guidelines
ax1.set_xlabel(r"$L_{\mathrm{xy}}$")
ax1.set_ylabel("Energy")

ax1.plot(
    box_sizes,
    refs,
    label=r"$\mathrm{Analytical}$",  # Assuming reference represents analytical
    marker="x",
    linestyle="solid",
    color="purple",
)
ax1.plot(
    box_sizes, elcs, marker="o", color="orange", label="Custom ELC", linestyle="dashed"
)

# Secondary Y-axis: Absolute Difference (Error)
ax2 = ax1.twinx()
error_color = "red"  # Distinct from energy lines
abs_diff = np.abs(np.array(refs) - np.array(elcs))

ax2.set_ylabel("Error", color=error_color)
ax2.plot(
    box_sizes,
    abs_diff,
    "s:",
    color=error_color,
    label=r"$\mathrm{|Analytical - Custom\ ELC|}$",
)
ax2.tick_params(axis="y", labelcolor=error_color, colors=error_color)
ax2.set_yscale("log")

# Combine labels into a single legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")

plt.grid(True, which="both", axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save using the mandated function
save_plot_with_timestamp(fig=fig)
plt.show()
