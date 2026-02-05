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
    """
    Calculates ELC energy from scratch based on elc.cpp logic.
    """
    # 1. System parameters
    box_l = np.array(system.box_l)
    prefactor = actor.prefactor 
    volume = box_l[0] * box_l[1] * box_l[2]
    
    # Correctly access particle properties as numpy arrays
    q = system.part.all().q
    pos = system.part.all().pos
    
    # shift = L/2 as defined in elc.cpp
    shift = box_l[2] / 2.0 

    # --- 2. Dipole Term (Yeh & Berkowitz) ---
    # According to dipole_energy() in elc.cpp: 
    # pref = prefactor * 2 * PI / volume
    dipole_pref = 2.0 * np.pi * prefactor / volume
    # gblcblk[2] = sum q_i (z_i - L/2)
    moment_z = np.sum(q * (pos[:, 2] - shift))
    # energy = 2 * pref * (moment_z^2)
    energy_dipole = 2.0 * dipole_pref * (moment_z**2)

    # --- 3. Z-Correction Term ---
    # Calculated in z_energy() in elc.cpp
    # For neutral systems without dielectric contrast, this term is 0.0
    energy_z = 0.0 

    # --- 4. Far-Field (Fourier) Terms ---
    energy_fourier = 0.0
    # In elc.cpp, this is determined by tune_far_cut()
    # We use a standard cutoff for the manual implementation
    far_cut = 5.0  
    far_cut2 = far_cut**2
    
    Lx, Ly, Lz = box_l
    ux = 2.0 * np.pi / Lx
    uy = 2.0 * np.pi / Ly
    
    max_p = int(np.ceil(far_cut * Lx / (2.0 * np.pi)))
    max_q = int(np.ceil(far_cut * Ly / (2.0 * np.pi)))

    # Replicating the PoQ and PQ energy loops from elc.cpp calc_energy()
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            
            omega2 = (p * ux)**2 + (q * uy)**2
            if omega2 > far_cut2:
                continue
                
            omega = np.sqrt(omega2)
            
            # pref_di logic from setup_PoQ and setup_PQ in elc.cpp:
            # PoQ (p=0 or q=0) uses 4*PI, PQ (p>0 and q>0) uses 8*PI
            if p == 0 or q == 0:
                fac = (4.0 * np.pi * prefactor / (Lx * Ly))
            else:
                fac = (8.0 * np.pi * prefactor / (Lx * Ly))
            
            # The exponential denominator: -pref_di / expm1(omega * Lz)
            fac /= (np.exp(omega * Lz) - 1.0)

            k_dot_rho = p * ux * pos[:, 0] + q * uy * pos[:, 1]
            
            # Complex exponential sums used for PoQ/PQ_energy in elc.cpp
            S_pos = np.sum(q * np.exp(omega * pos[:, 2]) * np.exp(1j * k_dot_rho))
            S_neg = np.sum(q * np.exp(-omega * pos[:, 2]) * np.exp(1j * k_dot_rho))
            
            # Replicating energy summation: (S_pos * conj(S_neg) + S_neg * conj(S_pos)) / omega
            term = (S_pos * np.conj(S_neg) + S_neg * np.conj(S_pos)).real
            energy_fourier += fac * term / omega

    # --- 5. Total Energy ---
    # FIX: Access energy via system.analysis, not the actor object directly
    p3m_energy = system.analysis.energy()['total']
    
    # elc.cpp returns 0.5 * energy_fourier for the frequency part
    return p3m_energy + energy_dipole + energy_z + (0.5 * energy_fourier)

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



