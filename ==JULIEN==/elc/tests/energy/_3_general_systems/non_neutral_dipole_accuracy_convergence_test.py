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

@pytest.fixture(scope="module")
def es_system():
    l_xy = 10.0
    l_z = 3.0
    system = espressomd.System(box_l=[l_xy, l_xy, l_z])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    
    yield system
    system.part.clear()

from elc.src.energy.third_party.get_ewald_energy_2d import direct_sum_energy, get_ewald_energy_2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
PREFACTOR = 1.7
@pytest.mark.parametrize("show_convergence_plot", [True]) 
def test_accuracy_convergence(es_system, show_convergence_plot):
    pw_errors = np.logspace(-6, -8, num=3)
    
    
    system = es_system
    gap_size = 1.0
    
    lx, ly, lz = system.box_l
    pos1, pos2 = get_rdm_constrained_points(lx, ly, lz-gap_size-1e-3)
    
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
        system.part.add(pos=pos2, q=+1.0)
        
        ana_energy = get_ewald_energy_2d(system, 100, prefactor=PREFACTOR)
        
        p3m = espressomd.electrostatics.P3M(prefactor=PREFACTOR, accuracy=pw_err, check_neutrality=False)
        system.electrostatics.solver = p3m
        
        prefactor, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(p3m, gap_size, pw_err, system)
        print(f"{pw_err=}: {e_recip=}, {e_3d=}, {e_non_neutral_corr}")
        assert prefactor == PREFACTOR
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
    assert slope > 0.5 and p_value < 0.05 # pyright: ignore[reportOperatorIssue]
    assert elc_errors[0] / min(elc_errors) > 50

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        
        # --- 1. Secondary Axis for Energy Contributions (Bars) ---
        ax2 = ax1.twinx()
        colors = ['#1abc9c', '#f1c40f', '#9b59b6']
        bottoms = np.zeros(len(pw_errors))

        # Track total heights to place labels at the top of the stack
        total_heights = np.zeros(len(pw_errors))
        for vals in contrib_data.values():
            total_heights += np.array(vals)

        # Width calculation in log-space
        bar_width = 0.2 * np.array(pw_errors) 

        for i, (label, vals) in enumerate(contrib_data.items()):
            bars = ax2.bar(pw_errors, vals, bottom=bottoms, width=bar_width, 
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
        
        plt.title('Non-Neutral Dipole Accuracy Convergence')
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, labels + bar_labels, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.invert_xaxis() 
        fig.tight_layout()
        plt.show()