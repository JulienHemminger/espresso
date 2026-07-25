import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy

# from src.elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
acc = 1e-6
gap_size = 8.0
box_l = 10.0


system = espressomd.System(box_l=[box_l, box_l, box_l + gap_size])
system.time_step = 0.01
system.cell_system.skin = 0.1

system.use_verlet_lists = True
system.periodicity = [True, True, True]

"""
* legacy is not symmetrical
* legacy, custom dont match
"""

params = {
    "lx": box_l,
    "ly": box_l,
    "lz": box_l + gap_size,
    "accuracy": acc,
    "pw_error": acc,
    "maxPWerror": acc,
    "prefactor": 1.0,
    "check_neutrality": False,
    "gap_size": gap_size,
    "delta_mid_bot": -1.0,
    "delta_mid_top": -1.0,
    "const_pot": True,
}
params["positions"] = [
    np.array([box_l / 2, box_l / 2, 0]),
    np.array([1.0, 2.0, 5.0]),
]
params["charges"] = [-2, +1]
particle_count = len(params["charges"])

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


sample_count = 30  # make even, to hide error spike at lz/2


z_pad = 2
z = np.linspace(z_pad, box_l - z_pad, sample_count)
legacy_energies = np.empty(sample_count)
custom_energies = np.empty(sample_count)
reference_energies = np.empty(sample_count)

for i in range(sample_count):
    # SETUP
    system.part.clear()
    system.box_l = [box_l, box_l, box_l + gap_size]
    for j in range(particle_count):
        system.part.add(id=j, pos=params["positions"][j], q=params["charges"][j])

    system.electrostatics.solver = elc

    # CALC
    system.part.by_id(0).pos = [box_l / 2, box_l / 2, z[i]]
    system.integrator.run(0)
    legacy_energies[i] = system.analysis.energy()["total"]

    custom_energies[i] = get_elcic_energy(system, params)["e_total"]

    reference_energies[i] = 0  # get_ewald_elcic_2d(params)

    print(
        f"legacy_energy={legacy_energies[i]}, custom_energy={custom_energies[i]}, reference_energy={reference_energies[i]}"
    )


fig, ax = plt.subplots()
linewidth = 3
ax.plot(
    z,
    legacy_energies,
    label="Legacy ELCIC",
    linestyle="--",
    marker="s",
    color="orange",
    linewidth=linewidth,
)
ax.plot(
    z,
    custom_energies,
    label="Custom ELCIC",
    linestyle="--",
    marker="v",
    color="blue",
    linewidth=linewidth,
)
# ax.plot(z, reference_energies, label="Reference", color="purple", linewidth=linewidth)
ax.set_xlabel("Particle 1: z position")
ax.set_ylabel("Energy")
ax.legend()

plt.tight_layout()
from src.common.plot_saving import save_plot_with_timestamp

save_plot_with_timestamp(fig)

plt.show()
