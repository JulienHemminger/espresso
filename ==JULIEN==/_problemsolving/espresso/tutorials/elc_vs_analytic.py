# testsuite/python/elc_vs_analytic.py
import sys

sys.path.insert(
    0, "/home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso/tests/python"
)


import espressomd
import espressomd.code_info
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

box_l = 200.0
system = espressomd.System(box_l=[box_l, box_l, box_l])
accuracy = 1e-7
check_accuracy = 1e-4
elc_gap = 75.0
system.time_step = 0.01
delta_mid_top = 0.0
delta_mid_bot = 39.0 / 41.0
distance = 1.0

minimum_distance_to_wall = 0.1
zPos = np.linspace(
    minimum_distance_to_wall,
    box_l - minimum_distance_to_wall - distance,
    6 if espressomd.code_info.build_type() == "Coverage" else 12,
)
q = np.arange(-5.0, 5.1, 2.5)


def get_legacy_forces_and_energy():
    _, p1, p2 = system.part.all()
    elc_forces = np.empty((len(q), len(zPos)))
    elc_energy = np.empty(elc_forces.shape)
    for chargeIndex, charge in enumerate(q):
        p1.q = charge
        p2.q = -charge
        for i, z in enumerate(zPos):
            pos = np.copy(p1.pos)
            p1.pos = [pos[0], pos[1], z]
            p2.pos = [pos[0], pos[1], z + distance]

            system.integrator.run(0)
            elc_forces[chargeIndex, i] = p1.f[2]
            elc_energy[chargeIndex, i] = system.analysis.energy()["total"]
    return elc_forces, elc_energy


def plot():
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    z_smooth = np.linspace(zPos.min(), zPos.max(), 200)
    charge_reshaped = prefactor * np.square(q.reshape(-1, 1))
    analytic_forces_smooth = charge_reshaped[0, 0] * (
        1 / distance**2
        + delta_mid_bot
        * (1 / np.square(2 * z_smooth) - 1 / np.square(2 * z_smooth + distance))
    )
    analytic_energy_smooth = charge_reshaped[0, 0] * (
        -1 / distance
        + delta_mid_bot
        * (
            1 / (4 * z_smooth)
            - 1 / (2 * z_smooth + distance)
            + 1 / (4 * (z_smooth + distance))
        )
    )

    # Force Plot
    ax1.plot(
        z_smooth, analytic_forces_smooth, "k-", label="Analytic Solution", alpha=0.7
    )
    ax1.plot(zPos, elc_forces[0], "ro", label="ELC Forces", markersize=5)
    ax1.set_xlabel("$z$ Position")
    ax1.set_ylabel("Force ($F_z$)")
    ax1.set_title("Force Comparison: ELC vs Analytic")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Energy Plot
    ax2.plot(
        z_smooth, analytic_energy_smooth, "k-", label="Analytic Solution", alpha=0.7
    )
    ax2.plot(zPos, elc_energy[0], "bs", label="ELC Energy", markersize=5)
    ax2.set_xlabel("$z$ Position")
    ax2.set_ylabel("Total Energy")
    ax2.set_title("Energy Comparison: ELC vs Analytic")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def get_analytical_forces_and_energy():
    charge_reshaped = prefactor * np.square(q.reshape(-1, 1))
    analytic_forces = charge_reshaped * (
        1 / distance**2
        + delta_mid_bot * (1 / np.square(2 * zPos) - 1 / np.square(2 * zPos + distance))
    )
    analytic_energy = charge_reshaped * (
        -1 / distance
        + delta_mid_bot
        * (1 / (4 * zPos) - 1 / (2 * zPos + distance) + 1 / (4 * (zPos + distance)))
    )
    return analytic_forces, analytic_energy


"""
Testing ELC against the analytic solution for an infinitely large
simulation box with dielectric contrast on the bottom of the box,
which can be calculated analytically with image charges.
"""
p3m_params = {"gpu": False}
rtol = 1e-7

system.box_l = [box_l, box_l, box_l + elc_gap]
system.cell_system.set_regular_decomposition(use_verlet_lists=True)
system.periodicity = [True, True, True]
# add a neutral particle before the two charges to make sure neutral
# particles aren't skipped in the multithreaded charge assignment loop

system.part.add(pos=[x * 0.5 for x in system.box_l], q=0.0)
system.part.add(pos=[x * 0.5 for x in system.box_l], q=q[0])
system.part.add(
    pos=np.array([x * 0.5 for x in system.box_l]) + np.array([0, 0, distance]),
    q=-q[0],
)
prefactor = 2.0
p3m = espressomd.electrostatics.P3M(
    prefactor=prefactor,
    accuracy=accuracy,
    mesh=[58, 58, 70],
    cao=4,
    **p3m_params,
)
elc = espressomd.electrostatics.ELC(
    actor=p3m,
    gap_size=elc_gap,
    maxPWerror=accuracy,
    delta_mid_bot=delta_mid_bot,
    delta_mid_top=delta_mid_top,
)
system.electrostatics.solver = elc
elc_forces, elc_energy = get_legacy_forces_and_energy()

# ANALYTIC SOLUTION
analytic_forces, analytic_energy = get_analytical_forces_and_energy()

plot()
np.testing.assert_allclose(elc_energy, analytic_energy, atol=1e-4)
np.testing.assert_allclose(elc_forces, analytic_forces, atol=1e-4, rtol=rtol)

# ./pypresso /home/main/Documents/Career/1_Studium/ESPRESSO/Cursor/espresso/==JULIEN==/_problemsolving/espresso/tutorials/elc_vs_analytic.py
