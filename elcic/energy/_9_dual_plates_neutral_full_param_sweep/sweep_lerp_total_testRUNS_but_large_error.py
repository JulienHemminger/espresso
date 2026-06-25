import espressomd
import numpy as np
from elcic.energy.custom_elcic.custom import get_elcic_energy
from elcic.energy._z_failed_legacy_contrib_parsing_experiment.get_legacy_contribs import get_legacy_contribs
from common.plotting.param_lerp_plot import run_lerp_plot

start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 4]),
        np.array([3, 2, 4]),
    ],  # legacy fails for part.z <= 3
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 15.0,  # legacy runs with: 14, 15, fails with 16, 20
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 30]),
        np.array([3, 2, 30]),
    ],  # legacy runs for part.z = 4, 24, fails for part.z = 3, 34, 39
}
end_params["lz"] = start_params["gap_size"] + 40


system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


def eval_custom_total(system, params):
    contribs = get_elcic_energy(system, params)
    return contribs["e_total"]

def eval_legacy_total(system, params):
    contribs = get_legacy_contribs(system, params)
    return contribs["E_total"]

# --- Mapping Configuration Table ---
plot_metrics = {
    ("Custom", (("color", "red"), ("marker", "x"), ("linestyle", "None"))): eval_custom_total,
    ("Legacy", (("color", "blue"), ("marker", "o"), ("linestyle", "None"))): eval_legacy_total
}

run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=6,
    plot_metrics=plot_metrics
)