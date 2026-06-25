import espressomd
import numpy as np
from elc.energy.get_custom_elc_energy import get_elc_energy_contribs
from common.legacy.energy import get_legacy_energy
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import (
    get_ewald_energy_2d
)
from common.plotting.param_lerp_energy_plot import run_lerp_plot
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 6])],
    "pw_error": 1e-8,
}
start_params["lz"] = start_params["gap_size"] + 40

end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10


# 3. Define the evaluation functions to match your new API signature: (system, params) -> float
def eval_analytical(sys, params):
    # Your analytical function only takes system in your original code
    return get_ewald_energy_2d(params)

def eval_legacy(sys, params):
    return get_legacy_energy(sys, params, timeout_duration_sec=600)

def eval_custom(sys, params):
    E_3d, E_dipole, E_recip, E_nonneutral = get_elc_energy_contribs(params["gap_size"], params["pw_error"], system, params["prefactor"])
    return E_3d + E_dipole + E_recip + E_nonneutral

# 4. Map metrics using the hashable tuple-of-tuples format for matplotlib configurations
metrics_to_plot = {
    ("Analytical", (("color", "blue"), ("marker", "o"), ("linewidth", 2))): eval_analytical,
    ("Legacy", (("color", "orange"), ("marker", "s"), ("linestyle", "--"), ("linewidth", 2))): eval_legacy,
    ("Custom", (("color", "red"), ("marker", "s"), ("linestyle", "--"), ("linewidth", 2))): eval_custom
}

# 5. Execute using the generalized plotting runner
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=6,
    plot_metrics=metrics_to_plot
)

