import espressomd
import espressomd.electrostatics
import numpy as np

def get_legacy_elc_energy(
    system, gap_size, pw_error, prefactor=1.0, delta_mid_top=None, delta_mid_bot=None
):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False,
        #cao=7, alpha=0.48017706031733637, mesh=np.array([34, 34, 34]), r_cut=7.275585142531804

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
