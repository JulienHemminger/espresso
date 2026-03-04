import numpy as np
import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from elc.src.common.get_positions import get_rdm_constrained_points
from elc.src.energy.get_elc_energy import get_elc_energy, get_elc_energy_contribs
from elc.src.energy.third_party.get_legacy_elc import get_legacy_elc_energy
import matplotlib.pyplot as plt
from scipy.stats import linregress

from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def get_legacy_forces(system, gap_size, pw_err) -> list[np.ndarray]:
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
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
    