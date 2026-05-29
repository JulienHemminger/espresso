import copy

import espressomd
import numpy as np
from elcic.energy.dual_plates.neutral.dipole.CUSTOM.custom import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_energy

from common.plotting.param_lerp_plot_2d import run_lerp_plot

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 20.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 14]), np.array([3, 2, 11])],    
}
start_params["lz"] = start_params["gap_size"] + 40

end_params = copy.deepcopy(start_params)



run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    get_custom_energy=get_elcic_energy, 
    get_analytical_energy=None,
    get_legacy_energy=get_legacy_energy,
    steps=1,
)

"""
[ELC] PQ_energy: omega=2.71270658963557, raw_sum=-1.21282432799041e-28, energy/omega=-4.47090124904862e-29
[ELC] calc_energy: PQ p=21 q=5 omega=2.71270658963557 contrib=-4.47090124904862e-29 energy_running=-0.00026039814858999
[ELC] setup_PQ p=21 q=6 omega=2.74453676945881: gblcblk[PQESSM]=-1.58548649739233e-82 gblcblk[PQESCM]=-2.18977074523363e-82 gblcblk[PQECSM]=-1.2596194543926e-81 gblcblk[PQECCM]=-1.7336700324958e-81 gblcblk[PQESSP]=7.74937624888977e-16 gblcblk[PQESCP]=4.87349495533433e-17 gblcblk[PQECSP]=-4.8876392973351e-17 gblcblk[PQECCP]=-3.23330494078585e-18
[ELC] PQ_energy: omega=2.74453676945881, raw_sum=-6.02105803261725e-29, energy/omega=-2.19383398306758e-29
[ELC] calc_energy: PQ p=21 q=6 omega=2.74453676945881 contrib=-2.19383398306758e-29 energy_running=-0.00026039814858999
[ELC] setup_PQ p=21 q=7 omega=2.78168471442291: gblcblk[PQESSM]=-2.21349491488791e-83 gblcblk[PQESCM]=-7.21451311996041e-84 gblcblk[PQECSM]=-1.7556324060072e-82 gblcblk[PQECCM]=-5.70425434933672e-83 gblcblk[PQESSP]=5.06867499641539e-16 gblcblk[PQESCP]=-9.66976970976076e-17 gblcblk[PQECSP]=-3.20062212836151e-17 gblcblk[PQECCP]=6.04574592762994e-18
[ELC] PQ_energy: omega=2.78168471442291, raw_sum=-2.65914145246579e-29, energy/omega=-9.55946386978461e-30
[ELC] calc_energy: PQ p=21 q=7 omega=2.78168471442291 contrib=-9.55946386978461e-30 energy_running=-0.00026039814858999
[ELC] setup_PQ p=22 q=1 omega=2.76745605479931: gblcblk[PQESSM]=2.15552655653192e-82 gblcblk[PQESCM]=2.96749902558624e-82 gblcblk[PQECSM]=1.78286072366872e-82 gblcblk[PQECCM]=2.45358140328765e-82 gblcblk[PQESSP]=1.36125780900153e-16 gblcblk[PQESCP]=5.30003933006793e-16 gblcblk[PQECSP]=-6.39678160184533e-17 gblcblk[PQECCP]=-2.49279640948391e-16
[ELC] PQ_energy: omega=2.76745605479931, raw_sum=-3.6373531122029e-29, energy/omega=-1.31433093793667e-29
[ELC] calc_energy: PQ p=22 q=1 omega=2.76745605479931 contrib=-1.31433093793667e-29 energy_running=-0.00026039814858999
[ELC] calc_energy: total (before *0.5)=-0.00026039814858999, final (0.5*total)=-0.000130199074294995
[ELC] === calc_energy() end ===
[ELC] long_range_energy: P3M_subtotal=-0.188864325133425, ELC_correction (calc_energy)=-0.000130199074294995, grand_total=-0.18899452420772
[ELC] === long_range_energy() end ===
DEBUG: Total particles: 2
DEBUG: Mask counts - Bot: 2, Top: 0, Mid: 0
DEBUG: E_l0: -0.1924, E_lt: -0.3868, E_pm1: -0.1924
DEBUG: Calculated E_near: -0.1934
custom_implementation_energy = -0.19340156375889017, error=0.0004476687437601212
ground_truth_energy = -0.19295389501513005
Figure successfully saved to: /home/main/lerp2d_2026-05-28_20-59-26.png
"""


"""
refac elc.cpp (more readable, etc)

map elc.cpp to python



"""

# setup system
# run legacy elcic
# extract ground truth contib values (E_near, E_far, E_total...)

# reset system
# run custom elcic
# plot custom contribs vs legacy contribs