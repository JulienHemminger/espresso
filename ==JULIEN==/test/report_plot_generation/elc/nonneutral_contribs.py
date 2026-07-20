import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
from src.elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d


def get_elc_energy_contribs(gap_size, pw_error, system, prefactor=1.0):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    # 1. 3D Periodic Energy from P3M
    system.electrostatics.solver = p3m
    E_3d = system.analysis.energy()["total"]

    lx, ly, lz = system.box_l
    particles = system.part.all()
    qs, (xs, ys, zs) = particles.q, particles.pos.T

    xi0 = np.sum(qs)
    xi1 = np.sum(qs * zs)
    xi2 = np.sum(qs * zs**2)
    volume = lx * ly * lz
    E_nonneutr = 2.0 * np.pi / volume * (-xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    E_dipole = 2.0 * np.pi / volume * xi1**2

    # 4. Reciprocal Space ELC Term
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))

    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()

    # Exclude the k=0 mode (handled by the real space and dipole terms)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P / lx) ** 2 + (Q / ly) ** 2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    # Particle-wise components
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Compute form factors (Chi) linearly
    def s_term(ez, c1, c2):
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Summing over the four combinations of sin/cos for the 2D Fourier transform
    chi = (
        s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy)
        + s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy)
        + s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy)
        + s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy)
    )

    # The reciprocal energy correction
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    E_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)

    return (E_3d, E_dipole, E_recip, E_nonneutr)


# 1. Initialize the system
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1]), np.array([1, 3, 5])],
    "pw_error": 1e-8,
}

t_values = np.linspace(0, 1, num=20)

analytical_results = []
E_3d_list = []
E_dipole_list = []
E_far_list = []
E_nonneutr_list = []
E_sum_list = []

# 2. Iterate and update particle positions
for t in t_values:
    system.part.clear()
    system.electrostatics.clear()

    # Update charges
    # Z=1.0000 | E_3d=2.8826 | E_dipole=0.0567 | E_recip=-2.0704 | E_nonneutr=-0.2274 | Sum=0.6415
    q0 = +5 + 3 * t
    q1 = -7 + 9 * t
    q2 = +2 - 2 * t

    system.part.add(pos=params["positions"][0], q=q0)
    system.part.add(pos=params["positions"][1], q=q1)
    system.part.add(pos=params["positions"][2], q=q2)

    # Calculate energies
    analytical_results.append(get_ewald_energy_2d(system))
    E_3d, E_dipole, E_recip, E_nonneutr = get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )

    E_3d_list.append(E_3d)
    E_dipole_list.append(E_dipole)
    E_far_list.append(E_recip)
    E_nonneutr_list.append(E_nonneutr)
    E_sum_list.append(E_3d + E_dipole + E_recip + E_nonneutr)

    print(
        f"Z={t:.4f} | E_3d={E_3d:.4f} | E_dipole={E_dipole:.4f} | E_recip={E_recip:.4f} | E_nonneutr={E_nonneutr:.4f} | Sum={E_sum_list[-1]:.4f}"
    )

from src.common.plot_saving import save_plot_with_timestamp

# Refactored plotting block
fig, ax1 = plt.subplots(figsize=(10, 6))

# All energy contributions on the primary axis
ax1.plot(t_values, E_3d_list, label=r"$\mathrm{E_{3D}}$", linestyle=":", color="cyan")
ax1.plot(
    t_values,
    E_sum_list,
    label=r"$\mathrm{E_{total}}$",
    linestyle="-",
    color="dodgerblue",
)
ax1.plot(t_values, E_far_list, label=r"$\mathrm{E_{far}}$", linestyle=":", color="lime")
ax1.plot(
    t_values,
    E_dipole_list,
    label=r"$\mathrm{E_{dipole}}$",
    linestyle=":",
    color="green",
)
ax1.plot(
    t_values,
    E_nonneutr_list,
    label=r"$\mathrm{E_{non-neutral}}$",
    linestyle=":",
    color="magenta",  # Using pink to distinguish from other defined colors
)
ax1.plot(
    t_values,
    analytical_results,
    label=r"$\mathrm{Ewald\ 2D}$",
    marker="o",
    linestyle="None",
    color="purple",
)

ax1.set_xlabel("Charge Parameter t")
ax1.set_ylabel("Energy")

# Single legend for all components
ax1.legend(loc="best")

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save using the mandated function
save_plot_with_timestamp(fig=fig)
plt.show()
