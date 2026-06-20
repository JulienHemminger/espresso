import espressomd
import espressomd.electrostatics
from common.generators.position_generator import get_rdm_constrained_points_np
from elc.energy.analytical.analytical_elc_energy import get_ewald_energy_2d

import matplotlib.pyplot as plt
import numpy as np
from src.common.has_downward_trend import has_downward_trend
from common.generators.position_generator import get_rdm_constrained_points_np

system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4
import espressomd
import espressomd.electrostatics
import numpy as np


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
    E_non_neutral_corr = 2.0 * np.pi / volume * (- xi0 * xi2 - (lz**2 / 12.0) * xi0**2)
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

    return (float(prefactor), float(E_recip), float(E_3d), float(E_dipole), float(E_non_neutral_corr))

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

        pref, E_recip, E_3d, E_dipole, E_nonneutr = get_elc_energy_contribs(
            gap_size, pw_err, system, prefactor
        )

        E_recip_final = pref * E_recip
        E_nonneutr_final = pref * E_nonneutr
        E_total = E_3d + E_nonneutr_final + E_recip_final + pref*E_dipole

        # Data collection
        elc_errors.append(abs(E_total - ana_energy))

        contrib_data["E_3d"].append(E_3d)
        contrib_data["E_dipole"].append(E_nonneutr_final)
        contrib_data["E_recip"].append(E_recip_final)

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

