import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

acc = 1e-6
gap_size = 8.0
box_l = 10.0


system = espressomd.System(box_l=[box_l, box_l, box_l + gap_size])
system.time_step = 0.01
system.cell_system.skin = 0.1

system.use_verlet_lists = True
system.periodicity = [True, True, True]

system.part.add(id=0, pos=(5.0, 5.0, 5.0), q=-9)
system.part.add(id=1, pos=(2.0, 2.0, 5.0), q=+1)
system.part.add(id=2, pos=(2.0, 5.0, 2.0), q=+1)
system.part.add(id=3, pos=(5.0, 2.0, 7.0), q=+1)

p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=acc, check_neutrality=False)

elc = espressomd.electrostatics.ELC(
    actor=p3m,
    gap_size=gap_size,
    maxPWerror=acc,
    delta_mid_bot=-1.0,
    delta_mid_top=-1.0,
    check_neutrality=False,
    const_pot=True,
)
system.electrostatics.solver = elc

sample_count = 100
z_step = 0.05
n_samples = sample_count + 1

z_pad = 2
z = np.linspace(z_pad, box_l - z_pad, n_samples)
z_forces = np.empty(n_samples)
energies = np.empty(n_samples)

for i in range(n_samples):
    system.part.by_id(0).pos = [2 * box_l, 2 * box_l, z[i]]
    system.integrator.run(0)
    z_forces[i] = system.part.by_id(0).f[2]
    energies[i] = system.analysis.energy()["coulomb"]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(z, z_forces, label="Force Z")
ax1.set_ylabel("Force")
ax1.legend()

ax2.plot(z, energies, label="Coulomb Energy", color="orange")
ax2.set_xlabel("z position")
ax2.set_ylabel("Energy")
ax2.legend()

plt.tight_layout()
plt.show()
