import espressomd
import espressomd.electrostatics
import numpy as np


def get_legacy_elc_energy(system, params_dict, fallback_return_value=-999):
    system.electrostatics.clear()

    try:
        gap_size = params_dict["gap_size"]
        pw_error = params_dict["pw_error"]
        prefactor = params_dict.get("prefactor", 1.0)
        delta_mid_top = params_dict.get("delta_mid_top")
        delta_mid_bot = params_dict.get("delta_mid_bot")

        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor,
            accuracy=pw_error,
            check_neutrality=False,
            verbose=True,
        )

        args = {
            "actor": p3m,
            "gap_size": gap_size,
            "maxPWerror": pw_error,
            "check_neutrality": False,
            "neutralize": False,
        }

        if delta_mid_top is not None:
            args["delta_mid_top"] = delta_mid_top
        if delta_mid_bot is not None:
            args["delta_mid_bot"] = delta_mid_bot
        args["const_pot"] = True

        elc_legacy = espressomd.electrostatics.ELC(**args)

        system.electrostatics.solver = elc_legacy
        system.integrator.run(0)

        print(system.analysis.energy())
        energy = system.analysis.energy()["total"]
        system.electrostatics.clear()
        return energy

    except Exception as e:
        print(f"--- ERROR: {e} ---")
        system.electrostatics.clear()
        return fallback_return_value


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

    energy = get_legacy_elc_energy(system, params, fallback_return_value=-999)
    print(f"{energy=}")


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




"""
Lets have elc.cpp with print statements like this: https://github.com/JulienHemminger/espresso/blob/03abcce3b48f5888df3f3a8e14e61b7264b83031/src/core/electrostatics/elc.cpp#L1265



E_return = ('coulomb', 0) + ('coulomb', 1) = -0.2725706666400204 - the actual returned value by "energy = system.analysis.energy()["total"]"

E_total = E_return_of(elc.cpp > ElectrostaticLayerCorrection::long_range_energy()) = ('coulomb', 1) = -0.2725706455416976
('coulomb', 0) = -2.1098322830323358e-08 (idk woher das kommt)
* Alex's vermutung ist dass ('coulomb', 0) eigentlich zu E_near_L0_L0 gehört. aber espresso addiert es erst später
"""
example_energy_dict = {
    "bonded": 0.0,
    "non_bonded_intra": 0.0,
    "non_bonded_inter": 0.0,
    ("dipolar", 1): 0.0,
    ("dipolar", 0): 0.0,
    ("coulomb", 0): -2.1098322830323358e-08,
    ("non_bonded_inter", 0, 0): 0.0,
    ("non_bonded", 0, 0): 0.0,
    ("coulomb", 1): -0.2725706455416976,
    "kinetic": 0.0,
    "total": -0.2725706666400204,
    "external_fields": 0.0,
    ("non_bonded_intra", 0, 0): 0.0,
    "kinetic_rot": 0.0,
    ("dpd", 0): 0.0,
    ("virtual_sites", 0): 0.0,
    "kinetic_lin": 0.0,
    "non_bonded": 0.0,
    "coulomb": -0.2725706666400204,
    "dipolar": 0.0,
    "virtual_sites": 0.0,
    "dpd": 0.0,
}
