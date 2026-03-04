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


def get_elc_forces(system, gap_size, pw_err) -> list[np.ndarray]:
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    
    f_3d = [p.f for p in system.part.all()]
    
    # todo: elc correction
    return f_3d
    