import numpy as np
import matplotlib.pyplot as plt
import espressomd
from common.legacy.energy import get_legacy_energy
from elc.energy._2_small_box_neutral_dipole.reference_solution.ewald2d import (
    get_ewald_energy_2d
)
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
    E_dipole_w_nonneutr_corr =  E_dipole# + 2.0 * np.pi / volume * (- xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

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

    #E_total = E_3d + (prefactor * E_dipole_w_nonneutr_corr) + (prefactor * E_recip)

    return (E_3d, E_dipole, E_recip)


from src.common.plot_saving import save_plot_with_timestamp

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
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 1]), np.array([3, 2, 15])],
    "pw_error": 1e-8,
}

# Define the Z range
lxy_values = np.linspace(20, 80, num=20)

analytical_results = []
E_3d_list = []
E_dipole_list = []
E_far_list = []
E_sum_list = []

# 2. Iterate and update particle positions
for l_xy in lxy_values:
    system.part.clear()
    system.electrostatics.clear()
    system.box_l = [l_xy, l_xy, params["lz"]]
    
    for i in range(len(params["positions"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    
    # Calculate energies
    analytical_results.append(get_ewald_energy_2d(system))
    E_3d, E_dipole, E_recip = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])

    E_dipole, E_recip = E_recip, E_dipole # TODO hack for plot
    
    E_3d_list.append(E_3d)
    E_dipole_list.append(E_dipole)
    E_far_list.append(E_recip)
    E_sum_list.append(E_3d + E_dipole + E_recip)

    print(f"Z={l_xy:.4f} | E_3d={E_3d:.4f} | E_dipole={E_dipole:.4f} | E_recip={E_recip:.4f} | Sum={E_sum_list[-1]:.4f}")

# 3. Plotting
fig = plt.figure(figsize=(10, 6))

# Plot components
plt.plot(lxy_values, E_3d_list, label='E_3d', linestyle=':', color='cyan')
plt.plot(lxy_values, E_dipole_list, label='E_dipole', linestyle=':', color='skyblue')
plt.plot(lxy_values, E_far_list, label='E_recip', linestyle=':', color='steelblue')
plt.plot(lxy_values, E_sum_list, label='Sum (E_3d + E_dipole)', linestyle='-', color='blue')
plt.plot(lxy_values, analytical_results, label='Analytical', marker='o', linestyle='None', color='red')

plt.xlabel("Box Size L_xy")
plt.ylabel("Energy")
plt.title("Energy Decomposition: E_3d, E_dipole, E_recip and Sum")
plt.legend()
plt.grid(True)

params_display = params.copy()
params_str = "Parameters:\n" + "\n".join([f"{k}: {v}" for k, v in params_display.items()])
plt.figtext(0.75, 0.5, params_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

save_plot_with_timestamp(fig)
plt.show()
# find params with nice contrib curves
# run n=20 plot

# TODO hack for better visuals: i swap E_dipole and E_recip