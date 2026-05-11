import espressomd
import numpy as np
from elcic.energy.shared.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "lz": 40.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([1, 1, 1]), np.array([2, 2, 2])],
    "pw_error": 1e-6,
}

end_params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 40.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([1, 1, 1]), np.array([2, 2, 2])],
    "pw_error": 1e-6,
}

run_lerp_plot(system, start_params, end_params, steps=2)

"""
LEGACY ELC EXCEPTIONS


* lxy=50, pw_error=1e-8: FAILS
    --- ERROR: tuning failed: an exception was thrown while benchmarking the integration loop (ELC tuning failed: maxPWerror too small) ---

* lxy=50, pw_error=1e-6: WORKS


* l_xy=10, pw_error=1e-8: WORKS


* l_xy=10, pw_error=1e-6: FAILS
    --- ERROR: tuning failed: an exception was thrown while benchmarking the integration loop (number of cells 5 is smaller than minimum 8: either interaction range is too large for the current skin (range=6.71226, half_local_box_l=[5, 5, 20]) or min_num_cells too large) ---
"""








"""
find good params to lerp over







* find the param range where legacy ELC works (no "Exception: while setting parameter 'box_l': ERROR: number of cells 4 is smaller than minimum 8: either interaction range is too large")
    * my custom_elcic needs to work on that param range too



* fix ana
    /home/main/Documents/Career/1_Studium/espresso/==JULIEN==/src/elcic/energy/single_plate/neutral/metallic/ana.py:77: RuntimeWarning: divide by zero encountered in divide
  return np.sum((q_a[:, None] * q_b[None, :] * erfc(alpha * r) / r)[diag_mask])

* fix custom
  File "/home/main/Documents/Career/1_Studium/espresso/==JULIEN==/src/elcic/energy/custom_elcic_energy.py", line 137, in get_elcic_energy_contribs
    system.box_l = [lx, ly, lz]
        Exception: while setting parameter 'box_l': ERROR: number of cells 4 is smaller than minimum 8: either interaction range is too large for the current skin (range=14.6008, half_local_box_l=[15, 15, 12.5]) or min_num_cells too large
    Exception: while setting parameter 'box_l': ERROR: P3M real-space cutoff too large for ELC w/ dielectric contrast
"""