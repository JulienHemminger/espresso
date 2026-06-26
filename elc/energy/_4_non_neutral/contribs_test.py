import espressomd
import espressomd.electrostatics
import numpy as np

from common.plotting.param_lerp_energy_plot import run_lerp_plot
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import (
    get_ewald_energy_2d,
)
from elc.energy.get_custom_elc_energy import get_elc_energy_contribs

# --- Modular Callback Wrappers for the Metrics Dictionary ---


def eval_e_3d(system, params):
    return get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )[0]


def eval_e_dipole(system, params):
    return get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )[1]


def eval_e_recip(system, params):
    return get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )[2]


def eval_e_nonneutr(system, params):
    return get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )[3]


def eval_e_sum(system, params):
    contribs = get_elc_energy_contribs(
        params["gap_size"], params["pw_error"], system, params["prefactor"]
    )
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
    (
        "Analytical",
        (("marker", "o"), ("linestyle", "None"), ("color", "red")),
    ): eval_analytical,
    ("E_dipole", (("linestyle", ":"), ("color", "skyblue"))): eval_e_dipole,
    ("E_non_neutr", (("linestyle", "-"), ("color", "lime"))): eval_e_nonneutr,
}

# --- Execution ---
# Set step count to your preference (e.g., 20 steps for a smooth interpolation curve)
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=20,
    plot_metrics=metrics_to_plot,
)
