import espressomd
import espressomd.electrostatics


def get_legacy_elc_energy(system, gap_size, pw_error):
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, accuracy=pw_error, check_neutrality=False, verbose=False
    )
    elc_legacy = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=gap_size,
        maxPWerror=pw_error,
        check_neutrality=False,
        neutralize=False,
    )
    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return system.analysis.energy()["total"]
