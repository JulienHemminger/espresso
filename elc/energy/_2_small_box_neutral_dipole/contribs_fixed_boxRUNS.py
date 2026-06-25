import numpy as np
import espressomd
import espressomd.electrostatics
from common.legacy.energy import get_legacy_energy
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import get_direct_sum_energy
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import get_ewald_energy_2d
# Assuming the newly generalized script is saved as param_lerp_plot.py in the same directory/path
from common.plotting.param_lerp_plot import run_lerp_plot

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




# --- Define the get_value wrapper metrics ---

def metric_E_3d(system, params):
    E_3d, _, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_3d

def metric_E_dipole(system, params):
    _, E_dipole, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_dipole

def metric_E_recip(system, params):
    _, _, E_recip = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_recip

def metric_E_sum(system, params):
    E_3d, E_dipole, E_recip = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_3d + E_dipole + E_recip

def metric_analytical(system, params):
    # return get_legacy_energy(system, params) #ana-custom match
    return get_ewald_energy_2d(params) # ana-custom match, but doesnt return values for edges
    # return get_direct_sum_energy(system) # ana-custom mismatch


# --- Setup ESPRESSOMD System Engine ---
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Static Configuration Elements
lx_val, ly_val, lz_val = 10.0, 10.0, 20.0
gap_size_val = 4.0
eps = 1e-1
z_min = 0 + eps
z_max = lz_val - gap_size_val - eps

# Defining start and end states to replicate the dynamic system trajectories
start_params = {
    "lx": lx_val, "ly": ly_val, "lz": lz_val,
    "gap_size": gap_size_val, "prefactor": 1.0, "pw_error": 1e-8,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6.0, 5.0, z_min]), np.array([3.0, 2.0, z_max])]
}

end_params = {
    "lx": lx_val, "ly": ly_val, "lz": lz_val,
    "gap_size": gap_size_val, "prefactor": 1.0, "pw_error": 1e-8,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6.0, 5.0, z_max]), np.array([3.0, 2.0, z_min])]
}

# --- Mapping Configuration Table ---
# Uses the immutable nested tuple framework required by the new plotting engine
plot_metrics = {
    ("E_3d", (("color", "cyan"), ("linestyle", ":"))): metric_E_3d,
    ("E_dipole", (("color", "skyblue"), ("linestyle", ":"))): metric_E_dipole,
    ("E_recip", (("color", "steelblue"), ("linestyle", ":"))): metric_E_recip,
    ("Sum (E_3d + E_dipole + E_recip)", (("color", "blue"), ("linestyle", "-"), ("linewidth", 2))): metric_E_sum,
    ("Analytical", (("color", "red"), ("marker", "o"), ("linestyle", "None"))): metric_analytical
}

# --- Run Plotting Engine ---
if __name__ == "__main__":
    run_lerp_plot(
        system=system,
        start_params=start_params,
        end_params=end_params,
        lerp_step_count=5, # Restoring original resolution parameters
        plot_metrics=plot_metrics
    )