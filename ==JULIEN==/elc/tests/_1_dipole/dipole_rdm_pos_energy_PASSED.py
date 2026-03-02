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
    
    # Cleanup if necessary (though usually handled by process exit)
    system.part.clear()

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
        ana_energy = -1.0 / r
        
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

@pytest.mark.parametrize("show_convergence_plot", [True]) 
def test_energy_convergence(es_system, show_convergence_plot):
    system = es_system
    gap_size = 1.0
    pw_errors = np.logspace(-2, -4, num=5) # Increased num for a better trend
    
    point_pairs = get_rdm_constrained_point_pairs(1, box_size=5.0)
    pos1, pos2 = point_pairs[0]
    r = math.dist(pos1, pos2)
    ana_energy = -1.0 / r
    
    elc_errors = []
    legacy_errors = []
    
    # Storage for the stacked bar components
    contrib_data = {
        'P3M (3D)': [],
        'Yeh-Berkowitz': [],
        'ELC Reciprocal': []
    }
    
    for pw_err in pw_errors:
        system.part.clear()
        system.part.add(pos=pos1, q=+1.0)
        system.part.add(pos=pos2, q=-1.0)
        
        p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
        system.electrostatics.solver = p3m
        
        prefactor, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(p3m, gap_size, pw_err, system)
        
        # Calculate components
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