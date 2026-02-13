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
import matplotlib.pyplot as plt

# 1. System Setup
box_size = 20.0
system = espressomd.System(box_l=[box_size, box_size, box_size * 3])
system.cell_system.skin = 0.4
system.time_step = 0.01

# Theoretical Madelung constant for 2D square lattice
madelung_2d_ref = -1.6155426267128247

# Data storage for plotting
ion_counts = []
errors = []

# 2. Loop through different spacings
spacings = [0.25, 0.5, 1, 2, 3]

print(f"{'Ions':>10} | {'Spacing':>10} | {'Calculated Energy':>18} | {'Error':>12}")
print("-" * 60)

for spacing in spacings:
    spacing = float(spacing)
    system.part.clear()

    ions_per_axis = int(box_size / spacing)
    ion_z_pos = system.box_l[2] / 2.0

    for i in range(ions_per_axis):
        for j in range(ions_per_axis):
            charge = (-1.0)**(i + j)
            system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

    # 3. Configure the Solvers
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, 
        accuracy=1e-6, 
    )

    elc = espressomd.electrostatics.ELC(
        actor=p3m, 
        gap_size=box_size, 
        maxPWerror=1e-6
    )
    system.electrostatics.solver = elc
    system.integrator.run(steps=0)

    # 4. Calculate Energy and Error
    elc_energy = system.analysis.energy()["total"]
    ion_count = len(system.part)
    elc_energy_per_ion = elc_energy / ion_count * (2.0 * spacing)
    error = np.abs(madelung_2d_ref - elc_energy_per_ion)

    # Store results
    ion_counts.append(ion_count)
    errors.append(error)

    print(f"{ion_count:10d} | {spacing:10.2f} | {elc_energy_per_ion:18.10f} | {error:12.10f}")

# 5. Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(ion_counts, errors, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)

# Formatting the plot
plt.xscale('log')  # Log scale for Number of Ions as requested
# Note: Since the error ranges from 10^-7 to 10^-2, a log scale for Y is also highly recommended:
plt.yscale('log') 

plt.xlabel('Number of Ions (Log Scale)', fontsize=12)
plt.ylabel('Absolute Error (Madelung Energy)', fontsize=12)
plt.title('ELC Validation: Error vs. System Size', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)

# Adding annotations for clarity
for i, spacing in enumerate(spacings):
    plt.annotate(f"s={spacing}", (ion_counts[i], errors[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('elc_madelung_validation_plot.png')
plt.show()

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