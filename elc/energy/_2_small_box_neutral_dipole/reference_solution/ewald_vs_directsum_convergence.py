import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.energy.analytical.analytical_elc_energy import (
    direct_sum_energy,
)
from elc.energy._2_small_box_neutral_dipole.reference_solution.ewald2d import (
    get_ewald_energy_2d
)

PREFACTOR = 1.7


def get_energies(n_values):

    # 1. System Setup
    l_xy = 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_xy, l_xy, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    # Reset system and add a dipole (1.0 and -1.0)
    system.part.clear()
    system.part.add(pos=[5.0, 5.0, 1.0], q=1.0)
    system.part.add(pos=[5.0, 5.0, 2.0], q=-1.0)

    ewald_energies = []
    direct_energies = []

    params = {
        "lx": l_xy,
        "ly": l_xy,
        "gap_size": 0,
        "prefactor": 1.0,
        "charges": [+1.0, -1.0],
        "positions": [np.array([5, 5, 1]), np.array([5, 5, 2])],
        "pw_error": 1e-8,
    }
    params["lz"] = params["gap_size"] + 3


    # 2. Calculation Loop
    for n in n_values:
        n_int = int(n)
        print(f"Processing n_max = {n_int}...")

        # Calculate both methods
        e_ewald = get_ewald_energy_2d(params, n_int)
        e_direct = direct_sum_energy(system, n_int, PREFACTOR)

        ewald_energies.append(e_ewald)
        direct_energies.append(e_direct)

    return ewald_energies, direct_energies


# 3. Execution and Plotting
n_range = np.geomspace(10, 100, num=5)
ewald_res, direct_res = get_energies(n_range)

plt.figure(figsize=(10, 6))

# Plotting both datasets
plt.plot(n_range, ewald_res, "o-", label="Ewald 2D Energy", markersize=8)
plt.plot(n_range, direct_res, "s--", label="Direct Sum Energy", alpha=0.7)

# Formatting for clarity
plt.xscale("log")
plt.xlabel("$n_{max}$ (Log Scale)")
plt.ylabel("Interaction Energy")
plt.title("Convergence Comparison: Ewald 2D vs. Direct Summation")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.show()
