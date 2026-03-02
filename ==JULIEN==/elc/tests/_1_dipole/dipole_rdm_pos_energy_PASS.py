import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from common.get_positions import get_rdm_constrained_point_pairs
from elc.src.get_elc_energy import get_elc_energy, get_elc_energy_contribs
from elc.src.get_legacy_elc import get_legacy_elc_energy
import pytest
import numpy as np
import math
import matplotlib.pyplot as plt
import espressomd
import espressomd.electrostatics
from scipy.stats import linregress
from common.get_positions import get_rdm_constrained_point_pairs
from elc.src.get_elc_energy import get_elc_energy
from elc.src.get_legacy_elc import get_legacy_elc_energy

@pytest.fixture(scope="module")
def es_system():
   
    l_xy = 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_xy, l_xy, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    yield system
    system.part.clear()
"""
@pytest.mark.parametrize("test_count", [5])  # Run 5 random pair tests
def test_elc_energy_accuracy(es_system, test_count):
    # Setup parameters
    pw_error = 1e-6
    gap_size = 1.0
    system = es_system
    l_z = system.box_l[2]
    l_xy = system.box_l[0]
    
    # Initialize P3M
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)
    
    # Generate points
    point_pairs = get_rdm_constrained_point_pairs(
        test_count, 
        box_size=min(l_xy, l_z - gap_size - 1e-3)
    )

    for pos1, pos2 in point_pairs:
        # 1. Reset particles
        system.part.clear()
        system.part.add(pos=pos1, q=+1.0)
        system.part.add(pos=pos2, q=-1.0)
        
        # 2. Calculate Reference (Analytical)
        r = math.dist(pos1, pos2)
        ana_energy = -1.0/r
        
        # 3. Calculate ELC and Legacy
        legacy_energy = get_legacy_elc_energy(p3m, gap_size, pw_error, system)
        elc_energy = float(get_elc_energy(p3m, gap_size, pw_error, system))
        
        # 4. Numerical Assertions
        # We expect the difference to be within the specified P3M error bound
        elc_diff = abs(elc_energy - ana_energy)
        legacy_diff = abs(legacy_energy - ana_energy)
        
        
        max_error = 1e3 * pw_error

        assert elc_diff < max_error, (
            f"ELC energy error {elc_diff} exceeded tolerance {max_error} at r={r}"
        )
        assert legacy_diff < max_error, (
            f"Legacy energy error {legacy_diff} exceeded tolerance {max_error} at r={r}"
        )
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc, erf

def ewald_energy_2d(positions, charges, dx, dy, n_max=100):
    eta=None
    n_real=n_max
    n_recip=n_max
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    N = len(q)
    A = dx * dy

    if eta is None:
        # Balance real/reciprocal convergence
        eta = np.sqrt(np.pi) / min(dx, dy)

    # Pair separation vectors: dr[a, b] = pos[a] - pos[b]
    dr = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    qq = q[:, None] * q[None, :]                     # (N, N)

    # ---- Real-space sum ----
    E_real = 0.0
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            R = np.array([nx * dx, ny * dy, 0.0])
            rvec = dr + R                             # (N, N, 3)
            dist = np.linalg.norm(rvec, axis=2)       # (N, N)

            if nx == 0 and ny == 0:
                np.fill_diagonal(dist, np.inf)        # exclude self

            contrib = qq * erfc(eta * dist) / dist
            E_real += np.sum(contrib)
    E_real *= 0.5

    # ---- Reciprocal-space sum (G != 0) ----
    gx0 = 2.0 * np.pi / dx
    gy0 = 2.0 * np.pi / dy
    drho = dr[:, :, :2]                               # in-plane (N, N, 2)
    dz = dr[:, :, 2]                                  # z-separation (N, N)

    E_recip = 0.0
    for mx in range(-n_recip, n_recip + 1):
        for my in range(-n_recip, n_recip + 1):
            if mx == 0 and my == 0:
                continue
            Gx = mx * gx0
            Gy = my * gy0
            G = np.sqrt(Gx**2 + Gy**2)

            phase = drho[:, :, 0] * Gx + drho[:, :, 1] * Gy  # (N, N)

            # h(G, dz) = exp(G*dz)*erfc(G/(2*eta) + eta*dz)
            #           + exp(-G*dz)*erfc(G/(2*eta) - eta*dz)
            arg_p = G / (2.0 * eta) + eta * dz
            arg_m = G / (2.0 * eta) - eta * dz
            h = np.exp(G * dz) * erfc(arg_p) + np.exp(-G * dz) * erfc(arg_m)

            E_recip += np.sum(qq * (np.pi / G) * h * np.cos(phase))

    E_recip /= (2.0 * A)

    # ---- Self-energy correction ----
    E_self = -(eta / np.sqrt(np.pi)) * np.sum(q ** 2)

    # ---- G = 0 term ----
    # Limit for |dz| -> 0:  |dz|*erf(eta*|dz|) + exp(-(eta*dz)^2)/(eta*sqrt(pi))
    #                      -> 1/(eta*sqrt(pi))
    adz = np.abs(dz)
    g0_terms = np.where(
        adz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        adz * erf(eta * adz) + np.exp(-(eta * adz) ** 2) / (eta * np.sqrt(np.pi)),
    )
    E_G0 = -np.pi / A * np.sum(qq * g0_terms)

    E_total = E_real + E_recip + E_self + E_G0
    return E_total


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

@pytest.mark.parametrize("show_convergence_plot", [True]) 
def test_energy_convergence(es_system, show_convergence_plot):
    system = es_system
    gap_size = 1.0
    pw_errors = np.logspace(-6, -10, num=4) # Increased num for a better trend
    
    point_pairs = get_rdm_constrained_point_pairs(1, box_size=5.0)
    pos1, pos2 = point_pairs[0]
    r = math.dist(pos1, pos2)
    
    elc_errors = []
    legacy_errors = []
    
    # Storage for the stacked bar components
    contrib_data = {
        'P3M (3D)': [],
        'Yeh-Berkowitz': [],
        'ELC Reciprocal': []
    }
    
    for pw_err in pw_errors:
        print(f"{pw_err=}: start")
        system.part.clear()
        system.part.add(pos=pos1, q=+1.0)
        system.part.add(pos=pos2, q=-1.0)
        
        p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
        system.electrostatics.solver = p3m
        
        prefactor, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(p3m, gap_size, pw_err, system)
        
        # Calculate components
        positions = system.part.all().pos  # Shape (N, 3)
        charges = system.part.all().q      # Shape (N,)

        # 2. Get box dimensions (assuming a rectangular box)
        dx = system.box_l[0]
        dy = system.box_l[1]

        # 3. Call the function
        ana_energy = ewald_energy_2d(positions, charges, dx, dy, n_max=100)
        print(f"{pw_err=}: {ana_energy=}")
        e_recip_final = prefactor * e_recip
        e_dipole_final = prefactor * e_non_neutral_corr
        elc_en = e_3d + e_dipole_final + e_recip_final
        
        elc_errors.append(abs(elc_en - ana_energy))
        legacy_errors.append(abs(get_legacy_elc_energy(p3m, gap_size, pw_err, system) - ana_energy))
        
        # Save for plotting
        contrib_data['P3M (3D)'].append(e_3d)
        contrib_data['Yeh-Berkowitz'].append(e_dipole_final)
        contrib_data['ELC Reciprocal'].append(e_recip_final)

    # --- Assertions (Keep your existing logic) ---
    slope, _, _, p_value, _ = linregress(np.log10(pw_errors), np.log10(elc_errors))
    #assert slope > 0.4 and p_value < 0.05
    #assert elc_errors[0] / min(elc_errors) > 50

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        
        # --- 1. Secondary Axis for Energy Contributions (Bars) ---
        ax2 = ax1.twinx()
        colors = ['#1abc9c', '#f1c40f', '#9b59b6']
        bottoms = np.zeros(len(pw_errors))
        
        # Width needs to be calculated in log-space to look consistent
        bar_width = 0.2 * np.array(pw_errors) 
        
        for i, (label, vals) in enumerate(contrib_data.items()):
            ax2.bar(pw_errors, vals, bottom=bottoms, width=bar_width, 
                    label=label, color=colors[i], alpha=0.3, edgecolor='grey')
            bottoms += np.array(vals)

        # --- 2. Primary Axis for Errors (Lines) ---
        ax1.loglog(pw_errors, elc_errors, 'o-', label='ELC Error', color='#2980b9', linewidth=2, zorder=5)
        ax1.loglog(pw_errors, legacy_errors, 's--', label='Legacy Error', color='#e67e22', alpha=0.7, zorder=4)
        ax1.loglog(pw_errors, pw_errors, 'k:', alpha=0.5, label='Target Accuracy (1:1)')

        # Formatting
        ax1.set_xlabel('Requested Accuracy (pw_error)')
        ax1.set_ylabel('Measured Error (Log Scale)', color='#2980b9')
        ax2.set_ylabel('Energy Component Value (Linear Scale)', color='#7f8c8d')
        
        plt.title('Convergence Analysis with Energy Decomposition')
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, labels + bar_labels, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.invert_xaxis() 
        fig.tight_layout()
        plt.show()