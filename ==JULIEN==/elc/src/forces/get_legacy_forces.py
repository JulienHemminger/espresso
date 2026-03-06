import espressomd
import espressomd.electrostatics
import numpy as np


def get_legacy_forces(system, gap_size, pw_err) -> list[np.ndarray]:
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, accuracy=pw_err, check_neutrality=False
    )
    elc_legacy = espressomd.electrostatics.ELC(
        actor=p3m,
        gap_size=gap_size,
        maxPWerror=pw_err,
        check_neutrality=False,
        neutralize=False,
    )
    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)

    return [p.f for p in system.part.all()]
