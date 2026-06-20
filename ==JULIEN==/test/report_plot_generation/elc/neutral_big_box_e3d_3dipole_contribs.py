# E-3d and E-corr contribs vs direct sum
# what do i lerp? part.z - something with interesting contribs

import numpy as np
import matplotlib.pyplot as plt
import espressomd
from src.elc.energy.legacy_elc_energy import get_legacy_energy
from src.elc.energy.analytical.large_box_direct_sum import get_direct_sum_energy as get_direct_sum_energy
import espressomd.electrostatics
from src.common.plot_saving import save_plot_with_timestamp


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


"""
lz =  4, gap_size = 3, eps = 1e-1
    Z=0.1000 | E_3d=0.0000 | E_dipole=-0.2296 | Sum=-0.2296
    Z=0.5000 | E_3d=0.0000 | E_dipole=-0.2335 | Sum=-0.2335
    Z=0.9000 | E_3d=0.0000 | E_dipole=-0.2296 | Sum=-0.2296

lz = 14, gap_size = 3, eps = 1e-1: WORSE
    Z=0.1000 | E_3d=0.0000 | E_dipole=-0.1174 | Sum=-0.1174
    Z=5.5000 | E_3d=0.0000 | E_dipole=-0.2335 | Sum=-0.2335
    Z=10.9000 | E_3d=0.0000 | E_dipole=-0.1174 | Sum=-0.1174

lz =  4, gap_size = 1, eps = 1e-1
    Z=0.1000 | E_3d=0.0000 | E_dipole=-0.1963 | Sum=-0.1963
    Z=1.5000 | E_3d=0.0000 | E_dipole=-0.2335 | Sum=-0.2335
    Z=2.9000 | E_3d=0.0000 | E_dipole=-0.1963 | Sum=-0.1963

lz =  1, gap_size = .5, eps = 1e-1
    Z=0.1000 | E_3d=0.0000 | E_dipole=-0.2329 | Sum=-0.2329
    Z=0.2500 | E_3d=0.0000 | E_dipole=-0.2335 | Sum=-0.2335
    Z=0.4000 | E_3d=0.0000 | E_dipole=-0.2329 | Sum=-0.2329


"""

"""
=== How to get E_recip = 0 ===
* Massive gap_size ("lz": 20.0, "gap_size": 19.0,): NO, E_recip=-0.002
* Massive gap_size ("lz": 100.0, "gap_size": 99.0,): NO, E_recip=-0.0017
* Loose pw_error=1e-2: NO, E_recip=0.02

* The Purely In-Plane Dipole (z1 = z2 = 0): NO,  E_dipole=0.0000 | E_recip=-0.0013
* The Purely Vertical Dipole (xy1 = xy2): NO, E_dipole=0.0012 | E_recip=0.0069 
"""


# 1. Initialize the system
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

params = {
    "lx": 80.0,
    "ly": 80.0,
    "lz": 20.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 0]), np.array([3, 2, 0])], # Z will be overwritten
    "pw_error": 1e-8,
}

# Define the Z range
eps = 1e-1
z_min = 0 + eps
z_max = params["lz"] - params["gap_size"] - eps
z_values = np.linspace(z_min, z_max, num=20)

analytical_results = []
E_3d_list = []
E_dipole_list = []
E_sum_list = []

# 2. Iterate and update particle positions
for z in z_values:
    system.part.clear()
    system.electrostatics.clear()
    
    # Update positions
    z1 = z
    z2 = (z_max - (z - z_min)) 
    
    pos1 = [params["positions"][0][0], params["positions"][0][1], z1]
    pos2 = [params["positions"][1][0], params["positions"][1][1], z2]
    
    system.part.add(pos=pos1, q=params["charges"][0])
    system.part.add(pos=pos2, q=params["charges"][1])
    
    # Calculate energies
    analytical_results.append(get_direct_sum_energy(system))
    E_3d, E_dipole, E_recip = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    
    E_3d_list.append(E_3d)
    E_dipole_list.append(E_dipole + E_recip)
    E_sum_list.append(E_3d + E_dipole + E_recip)

    print(f"Z={z:.4f} | E_3d={E_3d:.4f} | E_dipole={E_dipole:.4f} | E_recip={E_recip:.4f} | Sum={E_sum_list[-1]:.4f}")

# 3. Plotting
fig = plt.figure(figsize=(10, 6))

# Plot components
plt.plot(z_values, E_3d_list, label='E_3d', linestyle=':', color='cyan')
plt.plot(z_values, E_dipole_list, label='E_dipole', linestyle=':', color='skyblue')
plt.plot(z_values, E_sum_list, label='Sum (E_3d + E_dipole)', linestyle='-', color='blue')
plt.plot(z_values, analytical_results, label='Analytical', marker='o', linestyle='None', color='red')

plt.xlabel("Particle Z Position")
plt.ylabel("Energy")
plt.title("Energy Decomposition: E_3d, E_dipole, and Sum")
plt.legend()
plt.grid(True)

# Generate custom parameter string
params_display = params.copy()
# Format the positions string to show 'z' as a variable
params_display["positions"] = "[np.array([6, 5, z]), np.array([3, 2, z])]"

params_str = "Parameters:\n" + "\n".join([f"{k}: {v}" for k, v in params_display.items()])

# Place text
plt.figtext(0.75, 0.5, params_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

save_plot_with_timestamp(fig)
plt.show()


