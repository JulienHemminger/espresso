import espressomd
import espressomd.electrostatics


def get_legacy_elc_energy(
    system, gap_size, prefactor=1.0, pw_error=1e-6, delta_mid_top=None, delta_mid_bot=None
):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
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

    cond = delta_mid_top == -1 and delta_mid_bot == -1
    if cond: # ValueError: ELC with two parallel metallic boundaries requires the const_pot option
        args["const_pot"] = True

    elc_legacy = espressomd.electrostatics.ELC(**args)

    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return system.analysis.energy()["total"]
