import matplotlib.pyplot as plt
import numpy as np

# 1. Setup mock data
accuracy = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
x_labels = ["$10^{-4}$", "$10^{-5}$", "$10^{-6}$", "$10^{-7}$", "$10^{-8}$"]

# Mock data for stacked bars
f3d = [0.018] * 5
fdipole = [0.035] * 5
ffar = [0.002] * 5

# Mock data for the line plot
error_data = [2e-4, 1.2e-6, 2.4e-8, 1e-9, 5e-11]

# 2. Initialize plot
fig, ax1 = plt.subplots(figsize=(10, 7))

# 3. Create Stacked Bar Chart (on ax1)
width = 0.5
p1 = ax1.bar(
    x_labels, f3d, width, label="$F_{3D}$", color="#B2E2D7", edgecolor="gray", alpha=0.8
)
p2 = ax1.bar(
    x_labels,
    fdipole,
    width,
    bottom=f3d,
    label="$F_{dipole}$",
    color="#F7E8A6",
    edgecolor="gray",
    alpha=0.8,
)
p3 = ax1.bar(
    x_labels,
    ffar,
    width,
    bottom=np.array(f3d) + np.array(fdipole),
    label="$F_{far}$",
    color="#D9CCE3",
    edgecolor="gray",
    alpha=0.8,
)

# 4. Create Secondary Axis for the Line Plot
ax2 = ax1.twinx()
ax2.plot(
    x_labels,
    error_data,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="$|Ewald 2D - Custom ELC|$",
    color="#2171A8",
)

# 5. Styling
ax1.set_ylabel("Force Contribution Magnitude", fontsize=12)
ax1.set_xlabel("Requested Accuracy", fontsize=12)
ax2.set_ylabel("Error", fontsize=12, color="#2171A8")
ax2.set_yscale("log")

# Grid and Ticks
ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.tick_params(axis="y")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(
    lines + lines2, labels + labels2, loc="upper right", bbox_to_anchor=(1.25, 1)
)

plt.tight_layout()
plt.show()
