# %%
import sys
import os
# To allow me to write/run juptyer notebooks from VScode.
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))

# %%
import espressomd # type: ignore
import espressomd.electrostatics # type: ignore
import numpy as np
import math
from elc.get_legacy_elc import get_legacy_elc_energy

# %%
# TEST 1: Compare to analytical solution(energy, force) for a dipole, varying particle position, distance r
"""
from elc.tests.dipole_rdm_pos_energy_test import dipole_rdm_pos_energy_test
dipole_rdm_pos_energy_test(test_count=3)
"""
# %%
# TEST 2: Compare to analytical 2D Madelung energy of a crystal 
# Alex: Die Madelungen Energie ist halt die Energie pro Teilchen in einem unendlichen Kristall. Die konvergiert zu einer Konstanten, der Madelungen-Konstanten. Kann man analytisch zeigen, gibt in Espresso auch ein Testcase dazu. Kannst auch mal reinschauen

import espressomd
import espressomd.electrostatics
import numpy as np
import math

# 1. System Setup
box_size = 20.0
system = espressomd.System(box_l=[box_size, box_size, box_size * 3])
system.cell_system.skin = 0.4
system.time_step = 0.01

# 2. Create a 2D Square Lattice (Checkerboard)
ion_spacing = 3.0
for ion_spacing in [0.25, 0.5, 1, 2, 3]:
    ion_spacing = float(ion_spacing)
    system.part.clear()

    ions_per_axis = int(box_size / ion_spacing)
    ion_z_pos = system.box_l[2] / 2.0

    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0)**(i + j)
            system.part.add(pos=[i * ion_spacing, j * ion_spacing, ion_z_pos], q=charge)

    # 3. Configure the Solvers
    prefactor = 1.0
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, 
        accuracy=1e-6, 
    )

    # Initialize ELC with the P3M actor
    elc = espressomd.electrostatics.ELC(
        actor=p3m, 
        gap_size=box_size, 
        maxPWerror=1e-6
    )
    system.electrostatics.solver = elc
    system.integrator.run(steps=0)

    # 4. Calculate and Compare
    elc_energy = system.analysis.energy()["total"]
    ion_count = len(system.part)
    elc_energy_per_ion = elc_energy / ion_count * (2.0*ion_spacing)

    # Theoretical Madelung constant for 2D square lattice
    madelung_2d_ref = -1.6155426267128247

    print(f"--- ELC Validation ---")
    print(f"Number of ions: {ion_count} ({box_size=}, {ion_spacing=})")
    print(f"Calculated Energy per ion: {elc_energy_per_ion:.10f}")
    print(f"Reference Madelung Energy: {madelung_2d_ref:.10f}")
    print(f"error: {np.abs(madelung_2d_ref - elc_energy_per_ion):.10f}")

# %%
# TEST 5: Compare with the existing implementation of ELC for any different problems (generate system configurations randomly?, if possible compare all forces of evey particle + energy)

"""
--- ELC Validation ---
Number of ions: 1600 (box_size=20.0, ion_spacing=0.5)
Calculated Energy per ion: -1.6155432464
Reference Madelung Energy: -1.6155426267

--- ELC Validation ---
Number of ions: 400 (box_size=20.0, ion_spacing=1)
Calculated Energy per ion: -0.8077716102 x2
Reference Madelung Energy: -1.6155426267

--- ELC Validation ---
Number of ions: 100 (box_size=20.0, ion_spacing=2)
Calculated Energy per ion: -0.4038946962 x4
Reference Madelung Energy: -1.6155426267

--- ELC Validation ---
Number of ions: 36 (box_size=20.0, ion_spacing=3)
Calculated Energy per ion: -0.2573682771 x6
Reference Madelung Energy: -1.6155426267

"""