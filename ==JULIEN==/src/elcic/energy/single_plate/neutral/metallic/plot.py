import espressomd
import numpy as np
from elcic.energy.shared.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4 # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
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
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 6]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}
end_params["lz"] = end_params["gap_size"] + 10

run_lerp_plot(system, start_params, end_params, steps=10)

"""
Analytical Method
* fill in "hole", SKIP
    /home/main/Documents/Career/1_Studium/espresso/==JULIEN==/src/elcic/energy/single_plate/neutral/metallic/analytical.py:54: RuntimeWarning: divide by zero encountered in scalar divide
    E += q_a[i] * q_b[j] * erfc(alpha * r) / r

    * end_prams.pos1.z = 3: pass
    * end_prams.pos1.z = 4: error, hole
    * end_prams.pos1.z = 5: pass
    * end_prams.pos1.z = 6: pass

    
Custom Method
* 



"""