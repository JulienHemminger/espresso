import espressomd
import espressomd.electrostatics
import numpy as np
from elcic.energy.cpp_reverse_eng.legacy.get_legacy_contribs import get_legacy_contribs




system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # NEED to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)


def run_elc(params):
    system.electrostatics.clear()
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    legacy_contribs = get_legacy_contribs(system, params)
    print(f"{legacy_contribs=}")


params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 4]),
        np.array([3, 2, 4]),
    ],
}
params["lz"] = params["gap_size"] + 10


run_elc(params)
