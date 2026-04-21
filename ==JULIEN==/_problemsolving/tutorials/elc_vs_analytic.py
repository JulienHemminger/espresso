import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & Constants ---
BOX_L = 200.0
GAP_SIZE = 75.0
PREFACTOR = 2.0
Z_POS = 10.0
CENTER = BOX_L / 2.0
DISTANCE = 1.0
DELTA_MID_TOP = 0.0
DELTA_MID_BOT = 39.0 / 41.0

# Accuracies to test (logarithmically spaced)
accuracies = np.logspace(-2, -11, num=10)
force_diffs = []
energy_diffs = []


def calculate_analytic(z, dist):
    """Calculates analytic force and energy for q=1 at a specific z."""
    force = PREFACTOR * (
        1 / dist**2 + DELTA_MID_BOT * (1 / (2 * z) ** 2 - 1 / (2 * z + dist) ** 2)
    )
    energy = PREFACTOR * (
        -1 / dist
        + DELTA_MID_BOT * (1 / (4 * z) - 1 / (2 * z + dist) + 1 / (4 * (z + dist)))
    )
    return force, energy


# Pre-calculate analytic values
ana_force, ana_energy = calculate_analytic(Z_POS, DISTANCE)

# Initialize system
system = espressomd.System(box_l=[BOX_L, BOX_L, BOX_L + GAP_SIZE])
system.time_step = 0.01
system.cell_system.set_regular_decomposition(use_verlet_lists=True)
system.part.add(pos=[CENTER, CENTER, Z_POS], q=1.0)
system.part.add(pos=[CENTER, CENTER, Z_POS + DISTANCE], q=-1.0)


# Run simulations for different accuracies
for acc in accuracies:
    # Setup ELC with current accuracy
    p3m = espressomd.electrostatics.P3M(
        prefactor=PREFACTOR, accuracy=acc, check_neutrality=False
    )
    elc = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=GAP_SIZE,
        maxPWerror=acc,
        delta_mid_bot=DELTA_MID_BOT,
        delta_mid_top=DELTA_MID_TOP,
        neutralize=False,
    )
    system.electrostatics.solver = elc

    # Run a zero-step integration to calculate forces/energies
    system.integrator.run(0)

    # Collect results
    elc_force = system.part.by_id(0).f[2]
    elc_energy = system.analysis.energy()["total"]

    force_diffs.append(abs(elc_force - ana_force))
    energy_diffs.append(abs(elc_energy - ana_energy))


# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.loglog(accuracies, force_diffs, "o-", label="|Force ELC - Analytic|")
plt.loglog(accuracies, energy_diffs, "s--", label="|Energy ELC - Analytic|")

plt.xlabel("Accuracy (maxPWerror)")
plt.ylabel("Absolute Difference")
plt.title("Convergence of ELC Error vs. Requested Accuracy")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.gca().invert_xaxis()  # Show higher accuracy (smaller values) on the right
plt.savefig("elc_accuracy_convergence.png")
plt.show()
