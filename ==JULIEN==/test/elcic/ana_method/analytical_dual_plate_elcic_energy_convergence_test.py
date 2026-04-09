import pprint
import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elcic.energy.analytical.analytical_two_plate_elcic_energy import analytical_two_plate_elcic_energy
import time

system = espressomd.System(box_l=[1.0, 1.0, 1.0])
system.time_step = 0.01

# Parameters
params = {
    "lx": 100.0,
    "ly": 100.0,
    "lz": 20.0,
    "gap_size": 15.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1, -1],
    'pw_error': 1e-8,
    "positions": [np.array([2, 5, 3]), np.array([8, 3, 1])],
    "title": "Dual Plates, Both Metallic, Neutral"
}

# Setup System
system.box_l = [params["lx"], params["ly"], params["lz"] + params["gap_size"]]
for i, q in enumerate(params["charges"]):
    system.part.add(pos=params["positions"][i], q=q)

energies = []
N = 8 + 1
iterations = list([2*i for i in range(N)])

for i in iterations:
    start_time = time.perf_counter() # Start timer
    
    k_val = 2**i
    n_val = 10*i
    energy = analytical_two_plate_elcic_energy(system, params, k_max=k_val, n_max=n_val)
    energies.append(energy)
    
    elapsed = time.perf_counter() - start_time # Calculate duration
    print(f"Iteration {i}: energy({k_val=}, {n_val=}) = {energy} (took {elapsed:.4f} sec)")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(iterations, energies, "o-", color="tab:blue", label="Combined Convergence")

plt.xlabel("Iteration (i)")
plt.ylabel(r"Energy ($U_{total}$)")
plt.title(f"Convergence: $k_{{max}}=2^i, n_{{max}}=10i$")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Stats footer
stats_text = f"Final Energy: {energies[-1]:.6e}\nParams: {pprint.pformat(params)}"
plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=8, family="monospace", 
            bbox=dict(facecolor="white", alpha=0.8))

plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()

print("--- Energy Report ---")
print(f"Energies: {[round(float(e), 7) for e in energies]}")