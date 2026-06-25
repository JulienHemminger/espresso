import matplotlib.pyplot as plt
import numpy as np
import espressomd

from common.has_downward_trend import has_downward_trend
from common.generators.positions import get_rdm_constrained_points_np
from elc.energy._5_param_sweep_rdm_tests.custom_elc_energy_for_accuracy_convergence import get_elc_energy_contribs
from elc.energy._2_small_box_neutral_dipole.reference_solution.get_ewald2d_energy import get_ewald_energy_2d

def run_accuracy_convergence(
    system,
    params: dict,
    show_convergence_plot=True,
):
    # Extract parameters from dictionary
    gap_size = params["gap_size"]
    prefactor = params["prefactor"]
    charges = params["charges"]
    pw_target = params["pw_error"]
    
    # Setup
    pw_errors = np.logspace(-4, -8, num=5)
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    system.box_l = [lx, ly, lz]
    
    particle_count = len(charges)
    # Using provided positions from params or generating new ones based on constraints
    positions = params.get("positions", get_rdm_constrained_points_np(
        lx, ly, lz - gap_size - 1e-3, particle_count
    ))

    elc_errors = []
    contrib_data = {"P3M (3D)": [], "Yeh-Berkowitz": [], "ELC Reciprocal": []}

    for pw_err in pw_errors:
        system.part.clear()
        for i in range(particle_count):
            system.part.add(pos=positions[i], q=charges[i])

        # Use the passed params dict for the reference calculation
        ana_energy = get_ewald_energy_2d(params)

        pref, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(
            gap_size, pw_err, system, prefactor
        )

        e_recip_final = pref * e_recip
        e_dipole_final = pref * e_non_neutral_corr
        elc_en = e_3d + e_dipole_final + e_recip_final

        # Data collection
        elc_errors.append(abs(elc_en - ana_energy))
        contrib_data["P3M (3D)"].append(e_3d)
        contrib_data["Yeh-Berkowitz"].append(e_dipole_final)
        contrib_data["ELC Reciprocal"].append(e_recip_final)

    # --- Assertions ---
    #assert has_downward_trend(elc_errors)

    if show_convergence_plot:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()

        colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
        bottoms = np.zeros(len(pw_errors))
        bar_width = 0.2 * np.array(pw_errors)

        for i, (label, vals) in enumerate(contrib_data.items()):
            ax2.bar(pw_errors, vals, bottom=bottoms, width=bar_width, 
                    label=label, color=colors[i], alpha=0.3, edgecolor="grey")
            bottoms += np.array(vals)

        ax1.loglog(pw_errors, elc_errors, "o-", label="ELC Error", 
                   color="#2980b9", linewidth=2, zorder=5)
        ax1.loglog(pw_errors, pw_errors, "k:", alpha=0.5, label="Target Accuracy (1:1)")

        ax1.set_xlabel("Requested Accuracy (pw_error)")
        ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
        ax2.set_ylabel("Energy Component Value (Linear Scale)", color="#7f8c8d")
        plt.title("Energy Accuracy Convergence")

        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, labels + bar_labels, loc="upper left", bbox_to_anchor=(1.15, 1))
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.invert_xaxis()
        fig.tight_layout()
        plt.show()

# --- Example Usage ---
params = {
    "lx": 10.0,
    "ly": 10.0,
    "lz": 20.0,
    "gap_size": 2.1,
    "prefactor": 1.7,
    "charges": [+1.0, -1.0],
    "positions": [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 15.0])],
    "pw_error": 1e-8,
}

system = espressomd.System(box_l=[params["lx"], params["ly"], params["lz"]])
system.time_step = 0.01
system.cell_system.skin = 0.4

run_accuracy_convergence(system, params)
