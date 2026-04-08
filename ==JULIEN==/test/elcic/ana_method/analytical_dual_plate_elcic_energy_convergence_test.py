import pprint

import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from src.common.has_downward_trend import has_downward_trend
from elcic.energy.analytical.analytical_two_plate_elcic_energy import analytical_two_plate_elcic_energy

system = espressomd.System(box_l=[1.0, 1.0, 1.0])
system.time_step = 0.01

# Parameters
params = {
    "lx": 9.0,
    "ly": 12.0,
    "lz": 19.0,
    "gap_size": 15.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
    "charges": [+1, -1],
    'pw_error': 1e-8,
    "positions": [np.array([2, 5, 0]), np.array([8, 3, 0])],
    "title": "Dual Plates, Both Metallic, Neutral"
    }
positions = [np.array([7, 1, 0.1]), np.array([4, 5, 0.2])]

# Setup System
system.box_l = [params["lx"], params["ly"], params["lz"] + params["gap_size"]]
for i, q in enumerate(params["charges"]):
    system.part.add(pos=positions[i], q=q)

energies_pbc = []
energies_refl = []
N = 16 + 1
k_maxes = list([2**i for i in range(N)])
n_maxes = list([5*i for i in range(N)])

for i in range(N):
    print(f"Start for {i=}")
    energies_pbc.append(analytical_two_plate_elcic_energy(system, params, k_max=10*1, n_max=n_maxes[i]))
    energies_refl.append(analytical_two_plate_elcic_energy(system, params, k_max=k_maxes[i], n_max=10*1))
    

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(
    k_maxes, energies_pbc, "o-", color="tab:blue", label="PBC Convergence"
)
ax1.set_xscale("log", base=2)
ax1.set_xlabel(r"Number of PBC Images ($n_{max}$)")
ax1.set_ylabel(r"Energy ($U_{total}$)")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.plot(
    n_maxes,
    energies_refl,
    "s-",
    color="tab:red",
    label="Reflection Convergence",
)
ax2.set_xlabel(r"Reflection Steps ($k_{max}$)")
ax2.set_ylabel(r"Energy ($U_{total}$)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout(rect=(0, 0.15, 1, 1))
stats_text = f"Diff: {energies_pbc[-1] - energies_refl[-1]:.6e}\nParams: {pprint.pformat(params)}"
fig.text(
    0.5,
    0.02,
    stats_text,
    ha="center",
    fontsize=8,
    family="monospace",
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.show()

max_limit_diff = 9e-3
# Detailed status report
print("--- Energy Comparison Report ---")
print(f"energies_pbc={[round(float(e), 7) for e in energies_pbc]}")
print(f"energies_refl={[round(float(e), 7) for e in energies_refl]}")
print(
    f"Converge Towards Same Limit: {has_downward_trend(np.array(energies_pbc) - np.array(energies_refl))}"
)
print(
    f"Limit Difference: {np.abs(energies_refl[-1] - energies_pbc[-1])} <= {max_limit_diff=}: {np.abs(energies_refl[-1] - energies_pbc[-1]) <= max_limit_diff}"
)
print("--------------------------------")

