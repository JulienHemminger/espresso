import pprint
import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

# Assuming these local imports are in your python path
from src.common.has_downward_trend import has_downward_trend
from elcic.energy.analytical.analytical_two_plate_elcic_energy import analytical_two_plate_elcic_energy
from elcic.energy.analytical.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

def run_energy_convergence():
    # 1. Setup ESPResSo System
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    system.time_step = 0.01

    # 2. Parameters
    params = {
        "lx": 8.0,
        "ly": 12.0,
        "lz": 11.0,
        "gap_size": 9.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0, # for metallic it must: Δ = −1
        "charges": [+1, -1],
        'pw_error': 1e-8,
        "positions": [np.array([2, 5, 0.01]), np.array([8, 3, 0.02])]
    }

    # 3. Configure System Geometry
    system.box_l = [params["lx"], params["ly"], params["lz"] + params["gap_size"]]
    for i, q in enumerate(params["charges"]):
        system.part.add(pos=params["positions"][i], q=q)

    # 4. Calculation Loop
    energies_pbc = []
    energies_refl = []
    N = 9 + 1
    image_counts = [2**i for i in range(N)]
    reflection_counts = list(range(N))

    print("Running calculations...")
    for i in range(N):
        # PBC Convergence
        val_pbc = analytical_single_plate_2d_ewald_elcic_energy(
            params["positions"], params["charges"], system.box_l, 
            params["prefactor"], params["delta_mid_bot"], 
            k_max=10, n_real=image_counts[i]
        )
        energies_pbc.append(val_pbc)

        # Reflection Convergence
        val_refl = analytical_single_plate_2d_ewald_elcic_energy(
            params["positions"], params["charges"], system.box_l, 
            params["prefactor"], params["delta_mid_bot"], 
            k_max=reflection_counts[i], n_real=10
        )
        energies_refl.append(val_refl)
        print(f"Iteration {i} complete.")

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(image_counts, energies_pbc, "o-", color="tab:blue", label="PBC Convergence")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Number of PBC Images ($n_{max}$)")
    ax1.set_ylabel(r"Energy ($U_{total}$)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(reflection_counts, energies_refl, "s-", color="tab:red", label="Reflection Convergence")
    ax2.set_xlabel(r"Reflection Steps ($k_{max}$)")
    ax2.set_ylabel(r"Energy ($U_{total}$)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout(rect=(0, 0.15, 1, 1))
    stats_text = f"Diff: {energies_pbc[-1] - energies_refl[-1]:.6e}\nParams: {pprint.pformat(params)}"
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=8, family="monospace", bbox=dict(facecolor="white", alpha=0.8))

    # 6. Reporting and Logic Checks
    max_limit_diff = 9e-3
    limit_diff = np.abs(energies_refl[-1] - energies_pbc[-1])
    
    print("\n--- Energy Comparison Report ---")
    print(f"energies_pbc = {[round(float(e), 7) for e in energies_pbc]}")
    print(f"energies_refl = {[round(float(e), 7) for e in energies_refl]}")
    print(f"Converge Towards Same Limit: {has_downward_trend(np.array(energies_pbc) - np.array(energies_refl))}")
    print(f"Limit Difference: {limit_diff:.6e} <= {max_limit_diff}: {limit_diff <= max_limit_diff}")
    print("--------------------------------")

    # Show plot
    plt.show()


if __name__ == "__main__":
    run_energy_convergence()