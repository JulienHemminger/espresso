import espressomd
import espressomd.electrostatics


def get_legacy_elc_energy(actor, gap_size, pw_error, system):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor,
        gap_size=gap_size,
        maxPWerror=pw_error,
        check_neutrality=False,
        neutralize=False,
    )
    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return system.analysis.energy()["total"]


def get_legacy_elc_energy_new(system, gap_size, pw_error):
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, accuracy=pw_error, check_neutrality=False
    )
    return get_legacy_elc_energy(p3m, gap_size, pw_error, system)
