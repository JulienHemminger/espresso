import espressomd
import espressomd.electrostatics
import numpy as np

def get_legacy_elc_energy(system, params_dict, timeout_duration_sec=10, timeout_return_value=0.0):
   
    gap_size      = params_dict["gap_size"]
    pw_error      = params_dict["pw_error"]
    prefactor     = params_dict.get("prefactor", 1.0)
    delta_mid_top = params_dict.get("delta_mid_top")
    delta_mid_bot = params_dict.get("delta_mid_bot")
    
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False,
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
    if delta_mid_top == -1 and delta_mid_bot == -1:
        args["const_pot"] = True

    elc_legacy = espressomd.electrostatics.ELC(**args)

    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    #print(p3m.get_params())
    return system.analysis.energy()["total"]