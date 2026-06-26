import espressomd
import espressomd.electrostatics
import numpy as np

from common.plotting.param_lerp_energy_plot import run_lerp_plot
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import (
    get_direct_sum_energy as get_direct_sum_energy,
)
from elc.energy.get_custom_elc_energy import get_elc_energy_contribs

# Assuming 'run_lerp_plot', 'get_direct_sum_energy', and 'get_elc_energy_contribs'
# are already defined or imported here.

# --- 1. Set Up Constants & Bounds ---
lz_val = 20.0
gap_size_val = 10.0
eps = 1e-1
z_min = 0 + eps
z_max = lz_val - gap_size_val - eps

# --- 2. Define Start and End Parameter Dictionaries ---
start_params = {
    "lx": 80.0,
    "ly": 80.0,
    "lz": lz_val,
    "gap_size": gap_size_val,
    "prefactor": 1.0,
    "pw_error": 1e-8,
    "charges": [+1.0, -1.0],
    # At t=0: P0 is at z_min, P1 is at z_max
    "positions": [np.array([6.0, 5.0, z_min]), np.array([3.0, 2.0, z_max])],
}

end_params = {
    "lx": 80.0,
    "ly": 80.0,
    "lz": lz_val,
    "gap_size": gap_size_val,
    "prefactor": 1.0,
    "pw_error": 1e-8,
    "charges": [+1.0, -1.0],
    # At t=1: P0 is at z_max, P1 is at z_min
    "positions": [np.array([6.0, 5.0, z_max]), np.array([3.0, 2.0, z_min])],
}

# --- 3. Define Metric Evaluation Functions ---
# Each function matches the signature: (system, params) -> float


def eval_analytical(system, params):
    return get_direct_sum_energy(system)


def eval_e_3d(system, params):
    E_3d, _, _, _ = get_elc_energy_contribs(
        params["gap_size"],
        params["pw_error"],
        system,
        params["prefactor"],
    )
    return E_3d


def eval_e_dipole(system, params):
    _, E_dipole, E_recip, _ = get_elc_energy_contribs(
        params["gap_size"],
        params["pw_error"],
        system,
        params["prefactor"],
    )
    return E_dipole + E_recip


def eval_sum(system, params):
    E_3d, E_dipole, E_recip, _ = get_elc_energy_contribs(
        params["gap_size"],
        params["pw_error"],
        system,
        params["prefactor"],
    )
    return E_3d + E_dipole + E_recip


# --- 4. Define the Mapping Framework ---
# Map immutable nested configuration tuples to our evaluation functions
metrics_to_plot = {
    ("E_3d", (("color", "cyan"), ("linestyle", ":"))): eval_e_3d,
    ("E_dipole", (("color", "skyblue"), ("linestyle", ":"))): eval_e_dipole,
    ("Sum (E_3d + E_dipole)", (("color", "blue"), ("linestyle", "-"))): eval_sum,
    (
        "Analytical",
        (("color", "red"), ("marker", "o"), ("linestyle", "None")),
    ): eval_analytical,
}

# --- 5. Initialize System and Run ---
system = espressomd.System(box_l=[80, 80, 20])
system.time_step = 0.01
system.cell_system.skin = 0.4


# Fires off the generalized loop, validation step, plotting suite, and text overlay
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=6,
    plot_metrics=metrics_to_plot,
)
