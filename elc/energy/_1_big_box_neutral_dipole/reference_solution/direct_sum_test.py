import espressomd
import numpy as np

from common.legacy.energy import get_legacy_energy

# Assuming param_lerp_plot is inside your common/plotting path
from common.plotting.param_lerp_energy_plot import run_lerp_plot
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import (
    get_direct_sum_energy,
)

# 1. Initialize the system ONCE
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4

# 2. Define your Start and End Parameter spaces for interpolation
# We vary lx and ly from 20.0 to 80.0
start_params = {
    "lx": 20.0,
    "ly": 20.0,
    "lz": 20.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6.0, 5.0, 4.0]), np.array([3.0, 2.0, 1.0])],
    "pw_error": 1e-8,
}

end_params = start_params.copy()
end_params["lx"] = 80.0
end_params["ly"] = 80.0


# 3. Define the evaluation functions to match your new API signature: (system, params) -> float
def eval_analytical(sys, params):
    # Your analytical function only takes system in your original code
    return get_direct_sum_energy(sys)


def eval_legacy(sys, params):
    return get_legacy_energy(sys, params, timeout_duration_sec=600)


# 4. Map metrics using the hashable tuple-of-tuples format for matplotlib configurations
metrics_to_plot = {
    (
        "Analytical",
        (("color", "blue"), ("marker", "o"), ("linewidth", 2)),
    ): eval_analytical,
    (
        "Legacy",
        (("color", "orange"), ("marker", "s"), ("linestyle", "--"), ("linewidth", 2)),
    ): eval_legacy,
}

# 5. Execute using the generalized plotting runner
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=20,
    plot_metrics=metrics_to_plot,
)
