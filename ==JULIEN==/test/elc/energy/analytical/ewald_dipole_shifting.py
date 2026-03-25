import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from elc.energy.analytical_elc_energy import get_ewald_energy_2d

l_xy = 10.0
l_z = 3.0
gap_size = 1.0
system = espressomd.System(box_l=[l_xy, l_xy, l_z])
system.time_step = 0.01
system.cell_system.skin = 0.4

z_range = np.linspace(0, l_z - gap_size - 1e-3, num=10)
n_int = 100
ewald_energies = []

for z in z_range:
    system.part.clear()

    system.part.add(pos=[3.0, 5.0, z], q=+2.0)
    system.part.add(pos=[7.0, 5.0, z], q=-1.0)

    e_ewald = get_ewald_energy_2d(system, n_int)
    ewald_energies.append(e_ewald)


plt.figure(figsize=(10, 6))

plt.plot(
    z_range,
    ewald_energies,
    "-",
    color="tab:red",
    linewidth=2,
    label="Ewald 2D Energy ($n_{max}=100$)",
)

plt.xlabel("Dipole Height ($z$)")
plt.ylabel("Interaction Energy")
plt.title("2D Ewald Energy vs. Vertical Position (z)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
