import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from common.get_positions import get_rdm_constrained_points
from elc.src.get_elc_energy import get_elc_energy, get_elc_energy_contribs
from elc.src.third_party.get_legacy_elc import get_legacy_elc_energy
import matplotlib.pyplot as plt
from scipy.stats import linregress

from elc.src.third_party.get_ewald_energy_2d import get_ewald_energy_2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def test_accuracy_convergence():
    l_x, l_y = 10.0, 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_x, l_y, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    
    pw_err = 1e-2
    gap_size = 1.0
    
    pos1, pos2 = get_rdm_constrained_points(l_x, l_y, l_z-gap_size-1e-3)
   
    
    elc_errors = []
    legacy_errors = []
    
    # Storage for the stacked bar components
    contrib_data = {
        'P3M (3D)': [],
        'Yeh-Berkowitz': [],
        'ELC Reciprocal': []
    }
    
    system.part.clear()
    system.part.add(pos=pos1, q=+1.0)
    system.part.add(pos=pos2, q=-1.0)
    
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
    
    for p in system.part.all():
        print(f"{str(p.f)}")
        
        