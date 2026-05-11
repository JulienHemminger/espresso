import espressomd
import espressomd.electrostatics
import numpy as np

def get_legacy_elc_energy(system, params_dict, fallback_return_value=-999):
    system.electrostatics.clear()

    try:
        gap_size      = params_dict["gap_size"]
        pw_error      = params_dict["pw_error"]
        prefactor     = params_dict.get("prefactor", 1.0)
        delta_mid_top = params_dict.get("delta_mid_top")
        delta_mid_bot = params_dict.get("delta_mid_bot")
        
        p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=True,
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
        
        energy = system.analysis.energy()["total"]
        system.electrostatics.clear()
        return energy
    
    except Exception as e:
        print(f"--- ERROR: {e} ---")
        system.electrostatics.clear()
        return fallback_return_value

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01

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
    "lz": 10.0,
    "gap_size": 5.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 1])],
    "pw_error": 1e-8,
}


run_elc(params)
