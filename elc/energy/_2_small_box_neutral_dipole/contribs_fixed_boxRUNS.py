import numpy as np
import espressomd
import espressomd.electrostatics
from common.legacy.energy import get_legacy_energy
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import get_direct_sum_energy
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import get_ewald_energy_2d
# Assuming the newly generalized script is saved as param_lerp_plot.py in the same directory/path
from common.plotting.param_lerp_plot import run_lerp_plot
from elc.energy._1_big_box_neutral_dipole.custom1 import get_elc_energy_contribs

# --- Define the get_value wrapper metrics ---

def metric_E_3d(system, params):
    E_3d, _, _, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_3d

def metric_E_dipole(system, params):
    _, E_dipole, _, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_dipole

def metric_E_recip(system, params):
    _, _, E_recip, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_recip

def metric_E_sum(system, params):
    E_3d, E_dipole, E_recip, _ = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
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