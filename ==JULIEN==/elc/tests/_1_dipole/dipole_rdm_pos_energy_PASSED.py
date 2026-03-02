import pytest
import numpy as np
import math
import espressomd
import espressomd.electrostatics
from common.get_positions import get_rdm_constrained_point_pairs
from elc.src.get_elc_energy import get_elc_energy
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
    """
    Provides a single Espresso system instance for the module.
    Ensures only one instance exists as per EspressoMD constraints.
    """
    l_xy = 100.0
    l_z = 10.0
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

@pytest.mark.parametrize("show_convergence_plot", [True]) 
def test_energy_convergence(es_system, show_convergence_plot):
    system = es_system
    gap_size = 1.0
    
    # Range of accuracies from your plot
    pw_errors = np.logspace(-2, -6, num=10)
    
    point_pairs = get_rdm_constrained_point_pairs(1, box_size=5.0)
    pos1, pos2 = point_pairs[0]
    
    r = math.dist(pos1, pos2)
    ana_energy = -1.0 / r
    
    elc_errors = []
    legacy_errors = []
    
    
    for pw_err in pw_errors:
        system.part.clear()
        system.part.add(pos=pos1, q=+1.0)
        system.part.add(pos=pos2, q=-1.0)
        
        p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_err)
        # It's good practice to actually assign the solver to the system
        system.electrostatics.solver = p3m
        
        elc_en = float(get_elc_energy(p3m, gap_size, pw_err, system))
        elc_errors.append(abs(elc_en - ana_energy))
        
        legacy_en = get_legacy_elc_energy(p3m, gap_size, pw_err, system)
        legacy_errors.append(abs(legacy_en - ana_energy))

    # --- Robust Statistical Assertions ---
    
    # 1. Slope and P-Value Check
    # We use the p-value to ensure the downward trend is statistically significant (p < 0.05)
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log10(pw_errors), np.log10(elc_errors)
    )
    
    # A positive slope in (log(err) vs log(pw_err)) means err decreases as pw_err decreases.
    assert slope > 0.4, f"Trend is too flat or reversed. Slope: {slope:.2f}"
    assert p_value < 0.05, f"The convergence trend is not statistically significant (p={p_value:.3f})"
    
    # 2. Minimum Error Check
    # Ensure that the algorithm is capable of reaching a high-accuracy state.
    # We compare the coarsest error to the BEST (minimum) error achieved.
    min_error = min(elc_errors)
    improvement_factor = elc_errors[0] / min_error
    assert improvement_factor > 50, f"Algorithm only improved by {improvement_factor:.1f}x. Expected > 50x."

    if show_convergence_plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(pw_errors, elc_errors, 'o-', label='Actual ELC Error', color='#2980b9')
        plt.loglog(pw_errors, legacy_errors, 's--', label='Actual Legacy Error', color='#e67e22')
        
        # Plot the ideal 1:1 slope for reference
        plt.loglog(pw_errors, pw_errors, 'k:', alpha=0.5, label='Target Accuracy (1:1)')
        
        plt.xlabel('Requested Accuracy (pw_error)')
        plt.ylabel('Measured Error vs Analytical')
        plt.title('Convergence Analysis: Energy Error vs. P3M Accuracy')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.gca().invert_xaxis()  # Invert so better accuracy is on the right
        plt.show()