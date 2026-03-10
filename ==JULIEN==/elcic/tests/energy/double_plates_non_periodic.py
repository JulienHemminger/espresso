"""Tyagi_JCP_129.pdf

* system
    * lz=1, lx=ly=4, 10, 40
    * q=+1 at (0, 0, 1/4)
    * q=-1 at (0, 0, 3/4)
    * dielectic interfaces
        * at z = 0 and z = 1
        * eps_top=eps_bottom=eps_outside
        * eps_middle=1




* analytical solution
    * Coulomb’s law for an infinite open system and summing over the image charges,
    * f z = 41 +  k=0 − k 1 + 2k2 ,
        * where  = m − outside / m + outside
    * axpprox using cutoff simulation


* numerical solution
    * P3M, ELC of ESpresso


* measure F_z of q=+1 particle
"""

import espressomd.electrostatic_extensions
import espressomd.electrostatics
import numpy as np

pw_error = 1e-6
prefactor = 1.0

system = espressomd.System(box_l=[10, 10, 3])
system.time_step = 0.01

p3m = espressomd.electrostatics.P3M(
    prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
)
n_icc = 10
icc = espressomd.electrostatic_extensions.ICC(
    n_icc=n_icc,
    areas=np.zeros(n_icc),
    epsilons=np.zeros(n_icc),
    normals=np.full((n_icc, 3), [1, 0, 0]),
)


# Set the ICC line density and calculate the number of
# ICC particles according to the box size
box_l = 9.0
system.box_l = [box_l, box_l, 12.0]
nicc = 3  # linear density
nicc_per_electrode = nicc**2  # surface density
nicc_tot = 2 * nicc_per_electrode
iccArea = box_l**2 / nicc_per_electrode
L = box_l / nicc

# Lists to collect required parameters
iccNormals = []
iccAreas = []
iccSigmas = []
iccEpsilons = []

# Add the fixed ICC particles:
icc_type = 0
# Left electrode (normal [0, 0, 1])
for xi in range(nicc):
    for yi in range(nicc):
        system.part.add(
            pos=[L * xi, L * yi, 0.0], q=-0.0001, type=icc_type, fix=[True, True, True]
        )
iccNormals.extend([[0.0, 0.0, 1.0]] * nicc_per_electrode)

# Right electrode (normal [0, 0, -1])
for xi in range(nicc):
    for yi in range(nicc):
        system.part.add(
            pos=[L * xi, L * yi, box_l], q=0.0001, type=icc_type, fix=[True, True, True]
        )
iccNormals.extend([[0.0, 0.0, -1.0]] * nicc_per_electrode)

# Common area, sigma and metallic epsilon
iccAreas.extend([iccArea] * nicc_tot)
iccSigmas.extend([0.0] * nicc_tot)
iccEpsilons.extend([100000.0] * nicc_tot)
