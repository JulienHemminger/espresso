from common.plotting.param_lerp_plot import run_lerp_plot
import numpy as np
import matplotlib.pyplot as plt
import espressomd
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import (
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
    E_nonneutr = 2.0 * np.pi / volume * (- xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

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



# --- Modular Callback Wrappers for the Metrics Dictionary ---

def eval_e_3d(system, params):
    return get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])[0]

def eval_e_dipole(system, params):
    return get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])[1]

def eval_e_recip(system, params):
    return get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])[2]

def eval_e_nonneutr(system, params):
    return get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])[3]

def eval_e_sum(system, params):
    contribs = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return sum(contribs)

def eval_analytical(system, params):
    return get_ewald_energy_2d(params)


# --- Simulation Setup ---

system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Shared static variables
shared_config = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1]), np.array([1, 3, 5])],
    "pw_error": 1e-8,
}

# Explicitly mapping start (t=0) and end (t=1) charges for linear interpolation
start_params = {**shared_config, "charges": [5.0, -7.0, 2.0]}
end_params = {**shared_config, "charges": [8.0, 2.0, 0.0]}

# Build the metric evaluation dictionary using hashable nested tuple keys
metrics_to_plot = {
    ("E_3d", (("linestyle", ":"), ("color", "cyan"))): eval_e_3d,
    ("Sum", (("linestyle", "-"), ("color", "blue"), ("linewidth", 2))): eval_e_sum,
    ("E_recip", (("linestyle", ":"), ("color", "steelblue"))): eval_e_recip,
    ("Analytical", (("marker", "o"), ("linestyle", "None"), ("color", "red"))): eval_analytical,
    ("E_dipole", (("linestyle", ":"), ("color", "skyblue"))): eval_e_dipole,
    ("E_non_neutr", (("linestyle", "-"), ("color", "lime"))): eval_e_nonneutr
}

# --- Execution ---
# Set step count to your preference (e.g., 20 steps for a smooth interpolation curve)
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=20,
    plot_metrics=metrics_to_plot
)