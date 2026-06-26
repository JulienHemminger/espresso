import numpy as np
import espressomd
from common.legacy.energy import get_legacy_energy
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import get_direct_sum_energy
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import get_ewald_energy_2d

# Import your newly generalized plotting utility
# (Adjust this import path based on your actual file layout)
from common.plotting.param_lerp_energy_plot import run_lerp_plot

# 1. Initialize the system ONCE
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# 2. Define Start and End configurations for the interpolation
start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 10.0,
    "gap_size": 4.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}

# End parameters: Sweep lx and ly up to 30.0, keeping everything else the same
end_params = start_params.copy()
end_params["lx"] = 30.0
end_params["ly"] = 30.0

# 3. Define the custom metric wrappers matching signature: (system, params) -> float
def eval_direct_sum(sys, params):
    return get_direct_sum_energy(sys)

def eval_ewald2d(sys, params):
    # This specific function only requires the params dict
    return get_ewald_energy_2d(params)

def eval_legacy(sys, params):
    return get_legacy_energy(sys, params)

# 4. Map labels and plotting styles (using hashable immutable tuples) to the functions
metrics_to_plot = {
    ("direct_sum_results", (("marker", "o"), ("color", "blue"))): eval_direct_sum,
    ("ewald_results", (("marker", "o"), ("color", "orange"))): eval_ewald2d,
    ("legacy_results", (("marker", "s"), ("linestyle", "--"), ("color", "green"))): eval_legacy
}

# 5. Run the generalized loop and display/save the plot
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=20,
    plot_metrics=metrics_to_plot
)