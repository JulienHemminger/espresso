import espressomd
import espressomd.electrostatics
from common.generators.position_generator import get_rdm_constrained_points_np
from elc.energy.custom_elc_energy import get_elc_energy
from elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d

import matplotlib.pyplot as plt
import numpy as np
from src.common.has_downward_trend import has_downward_trend
from common.generators.position_generator import get_rdm_constrained_points_np
from elc.energy.custom_elc_energy import get_elc_energy_contribs
from elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d

system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4


def run_accuracy_convergence(
    system,
    prefactor=1.0,
    gap_size=1.0,
    charges=[+1.0, -1.0],
    show_convergence_plot=True,
):
    # Setup parameters based on system type
    pw_errors = np.logspace(-4, -8, num=5)

    title = "Energy Accuracy Convergence"

    lx, ly, lz = system.box_l

    particle_count = len(charges)
    positions = get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count
    )

    elc_errors = []
    contrib_data = {"E_3d": [], "E_dipole": [], "E_recip": []}

    for pw_err in pw_errors:
        system.part.clear()
        for i in range(particle_count):
            system.part.add(pos=positions[i], q=charges[i])

        ana_energy = get_ewald_energy_2d(system, n_max=100, prefactor=prefactor)

        pref, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(
            gap_size, pw_err, system, prefactor
        )

        e_recip_final = pref * e_recip
        e_dipole_final = pref * e_non_neutral_corr
        elc_en = e_3d + e_dipole_final + e_recip_final

        # Data collection
        elc_errors.append(abs(elc_en - ana_energy))

        contrib_data["E_3d"].append(e_3d)
        contrib_data["E_dipole"].append(e_dipole_final)
        contrib_data["E_recip"].append(e_recip_final)

    # --- Assertions ---
    assert has_downward_trend(elc_errors)

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()

        colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
        bottoms = np.zeros(len(pw_errors))
        bar_width = 0.2 * np.array(pw_errors)

        # 1. Secondary Axis: Energy Contributions (Stacked Bars)
        for i, (label, vals) in enumerate(contrib_data.items()):
            ax2.bar(
                pw_errors,
                vals,
                bottom=bottoms,
                width=bar_width,
                label=label,
                color=colors[i],
                alpha=0.3,
                edgecolor="grey",
            )
            bottoms += np.array(vals)

        # 2. Primary Axis: Errors (Lines)
        ax1.loglog(
            pw_errors,
            elc_errors,
            "o-",
            label="ELC Error",
            color="#2980b9",
            linewidth=2,
            zorder=5,
        )

        ax1.loglog(pw_errors, pw_errors, "k:", alpha=0.5, label="Target Accuracy (1:1)")

        # Formatting
        ax1.set_xlabel("Requested Accuracy (pw_error)")
        ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
        ax2.set_ylabel("Energy Component Value (Linear Scale)", color="#7f8c8d")
        plt.title(title)

        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + bars,
            labels + bar_labels,
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
        )

        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.invert_xaxis()
        fig.tight_layout()
        plt.show()


run_accuracy_convergence(
    system, prefactor=1.7, gap_size=2.1, charges=[-0.6, +1.5, -0.9]
)  # PASSED


"""

import numpy as np
import matplotlib.pyplot as plt
import espressomd
from src.elc.energy.analytical.large_box_direct_sum import get_direct_sum_energy
import espressomd.electrostatics


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

    E_dipole = 2.0 * np.pi / volume * xi1**2
    E_nonneutr_corr = 2.0 * np.pi / volume * (- xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

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


    return (E_3d, E_dipole, E_recip, E_nonneutr_corr)




# 1. Initialize the system
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 20.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1]), np.array([1, 3, 5])],
    "pw_error": 1e-8,
}

t_values = np.linspace(0, 1, num=3)

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
    q0 = +1+2*t
    q1 = -2+t
    q2 = +1-t
    
    system.part.add(pos=params["positions"][0], q=q0)
    system.part.add(pos=params["positions"][1], q=q1)
    system.part.add(pos=params["positions"][2], q=q2)
    
    # Calculate energies
    analytical_results.append(get_direct_sum_energy(system))
    E_3d, E_dipole, E_recip, E_nonneutr = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    
    E_3d_list.append(E_3d)
    E_dipole_list.append(E_dipole)
    E_far_list.append(E_recip)
    E_nonneutr_list.append(E_nonneutr)
    E_sum_list.append(E_3d + E_dipole + E_recip + E_nonneutr)

    print(f"Z={t:.4f} | E_3d={E_3d:.4f} | E_dipole={E_dipole:.4f} | E_recip={E_recip:.4f} | E_nonneutr={E_nonneutr:.4f} | Sum={E_sum_list[-1]:.4f}")

# 3. Plotting
plt.figure(figsize=(10, 6))

# Plot components
plt.plot(t_values, E_3d_list, label='E_3d', linestyle=':', color='cyan')
plt.plot(t_values, E_dipole_list, label='E_dipole', linestyle=':', color='skyblue')
plt.plot(t_values, E_far_list, label='E_recip', linestyle=':', color='steelblue')
plt.plot(t_values, E_nonneutr_list, label='E_nonneutr', linestyle='-', color='lime')
plt.plot(t_values, E_sum_list, label='Sum (E_3d + E_dipole)', linestyle='-', color='blue')
plt.plot(t_values, analytical_results, label='Analytical', marker='o', linestyle='None', color='red')

plt.xlabel("Particle Z Position")
plt.ylabel("Energy")
plt.title("Energy Decomposition: E_3d, E_dipole, and Sum")
plt.legend()
plt.grid(True)

params_display = params.copy()
params_display["charges"] = "[q1 = 1+t, q2 = 2-t, q3 = 4-3t]"
params_str = "Parameters:\n" + "\n".join([f"{k}: {v}" for k, v in params_display.items()])

# Place text
plt.figtext(0.75, 0.5, params_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

Instead of lerping particle.pos.z i want to lerp over particle.q.


x-axis lerp value t

have three particles with fixed positions, but charges
    q1 = 1+t
    q2 = 2-t
    q3 = 4-3t
"""
