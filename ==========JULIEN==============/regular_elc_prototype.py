# %%
import sys
import os
# To allow me to write/run juptyer notebooks from VScode.
espresso_build_path = "/home/main/Documents/Career/1_Studium/espresso/build"
sys.path.insert(0, os.path.join(espresso_build_path, "src", "python"))

# %%
import espressomd # type: ignore
import espressomd.electrostatics # type: ignore

# --- System Setup ---
box_l = 10.0
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = 0.01
system.cell_system.skin = 0.4

# Add Particles (keeping them within the non-gap region)
system.part.add(pos=[5.0, 5.0, 1.0], q=1.0)
system.part.add(pos=[5.0, 5.0, 7.0], q=-1.0)

# Initialize P3M deterministically
p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=1e-3, mesh=[32, 32, 32], cao=3, alpha=0.35, r_cut=4.5)

# Parameters for both methods
gap_size = 2.0
pw_error = 1e-3

# %%
import numpy as np

def get_newer_ELC_energy(actor, gap_size, pw_error):
    return 0.0

def get_legacy_ELC_energy(actor, gap_size, pw_error):
    elc_legacy = espressomd.electrostatics.ELC(
        actor=actor, 
        gap_size=gap_size, 
        maxPWerror=pw_error
    )
    system.electrostatics.solver = elc_legacy
    return system.analysis.energy()['total']

# %%
# --- Summary Comparison ---
legacy_elc = get_legacy_ELC_energy(p3m, gap_size, pw_error) # -0.023799
newer_elc = get_newer_ELC_energy(p3m, gap_size, pw_error)

print("\n" + "="*30)
print(f"{'Method':<15} | {'Energy':<15}")
print("-" * 30)
print(f"{'Legacy ELC':<15} | {legacy_elc:<15.6f}")
print(f"{'pyELC':<15} | {newer_elc:<15.6f}")
print("-" * 30)
print(f"Difference: {abs(legacy_elc - newer_elc):.6f}")
print("="*30)



