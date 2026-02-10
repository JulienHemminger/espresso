# %%
import sys
import os
# To allow me to write/run juptyer notebooks from VScode.
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))

# %%
import espressomd # type: ignore
import espressomd.electrostatics # type: ignore

# %%
def get_legacy_elc_energy(actor, gap_size, pw_error, system):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error
    )
    system.electrostatics.solver = elc_legacy
    return system.analysis.energy()['total']

def get_legacy_elc_force(actor, gap_size, pw_error, system):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error
    )
    system.electrostatics.solver = elc_legacy
    # espressomd.html#espressomd.particle_data.ParticleHandle.f
    return system.part.by_id(0).f
