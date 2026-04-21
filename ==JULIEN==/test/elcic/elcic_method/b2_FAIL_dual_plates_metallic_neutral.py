import matplotlib.pyplot as plt
import numpy as np
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs
import espressomd


from elc.energy.legacy_elc_energy import get_legacy_elc_energy



def run_elcic_diagnostic(system, z_pos_count, params):
    prefactor = params["prefactor"]
    pw_error = params["pw_error"]
    delta_mid_top = params["delta_mid_top"]
    delta_mid_bot = params["delta_mid_bot"]
    lx, ly, lz = params["lx"], params["ly"], params["lz"]
    gap = params["gap_size"]
    eps = 0.5

    system.box_l = [lx, ly, lz]
    z_range = np.linspace(eps, lz - gap - eps, num=z_pos_count)

    # Expanded data structure to capture sub-components
    data = {
        "z": [],
        "custom_total": [],
        "legacy_total": [],
        "e_near": [],
        "e_far": [],
        "l0": {"total": [], "e_3d": [], "e_elc": []},
        "pm1": {"total": [], "e_3d": [], "e_elc": []},
        "lt": {"total": [], "e_3d": [], "e_elc": []},
    }

    for z in z_range:
        system.part.clear()
        for i in range(len(params["charges"])):
            orig_pos = params["positions"][i]
            system.part.add(pos=[orig_pos[0], orig_pos[1], z], q=params["charges"][i])

        # 1. Custom ELCIC Breakdown
        contribs = get_elcic_energy_contribs(
            system, gap, pw_error, prefactor, delta_mid_bot, delta_mid_top
        )

        # 2. Legacy ELC
        e_legacy = get_legacy_elc_energy(
            system, gap, prefactor, pw_error, delta_mid_top, delta_mid_bot
        )

        data["z"].append(z)
        data["custom_total"].append(contribs["e_near"] + contribs["e_far"])
        data["legacy_total"].append(e_legacy)
        data["e_near"].append(contribs["e_near"])
        data["e_far"].append(contribs["e_far"])

        # Collect sub-components
        for key in ["l0", "pm1", "lt"]:
            data[key]["total"].append(contribs[key]["total"])
            data[key]["e_3d"].append(contribs[key]["e_3d"])
            data[key]["e_elc"].append(contribs[key]["e_elc"])

    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    z_vals = np.array(data["z"])

    # Plot 1: High-level Comparison + Error
    axes[0].plot(z_vals, data["custom_total"], "k-", lw=2, label="Total Custom ELCIC")
    axes[0].plot(z_vals, data["legacy_total"], "m--", lw=2, label="Total Legacy ELCIC")
    axes[0].plot(z_vals, data["e_near"], "r:", label="Custom: Near Field")
    axes[0].plot(z_vals, data["e_far"], "g:", label="Custom: Far Field")
    axes[0].set_title("Energy Comparison & Accuracy")
    axes[0].set_ylabel("Energy")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Secondary axis for error
    ax1_err = axes[0].twinx()
    error = np.array(data["legacy_total"]) - np.array(data["custom_total"])
    ax1_err.plot(z_vals, error, "b-", alpha=0.6, label="Legacy - Custom (Error)")
    ax1_err.set_ylabel("Error (Legacy - Custom)", color="b")
    ax1_err.tick_params(axis="y", labelcolor="b")
    ax1_err.legend(loc="upper right")

    # Component Plot Helper
    def plot_component_breakdown(ax, data_dict, title):
        ax.plot(z_vals, data_dict["total"], "k-", lw=1.5, label="Total")
        ax.plot(z_vals, data_dict["e_3d"], "r:", label="$e_{3d}$")
        ax.plot(z_vals, data_dict["e_elc"], "g:", label="$e_{elc}$")
        ax.set_title(title)
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: L0 Components
    plot_component_breakdown(axes[1], data["l0"], "$L_{0}$ Components (Real Space)")

    # Plot 3: PM1 Components
    plot_component_breakdown(
        axes[2], data["pm1"], "$L_{\pm 1}$ Components (First Images)"
    )

    # Plot 4: LT Components
    plot_component_breakdown(axes[3], data["lt"], "$L_{t}$ Components (Real + Images)")
    axes[3].set_xlabel("z-position")

    # Parameters textbox
    params_str = "\n".join([f"{k}: {v}" for k, v in params.items() if k != "positions"])
    fig.text(
        0.85,
        0.5,
        f"Parameters:\n{'-' * 15}\n{params_str}",
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.subplots_adjust(right=0.83, hspace=0.3)
    plt.show()


# To use:
system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params = {
    "lx": 8.0,
    "ly": 8.0,
    "lz": 11.0,
    "gap_size": 9.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.0,
    "delta_mid_bot": 0.3,  # for metallic it must: Δ = −1
    "charges": [+1, -1],
    "pw_error": 1e-6,
    "positions": [np.array([6, 2, 1]), np.array([0, 0, 2])],
    "title": "Contrib Debug, Neutral",
}
run_elcic_diagnostic(system, z_pos_count=16, params=params)

"""
* fix one params-dict to debug elcic.py on
    * am besten was wo es deutliche fehler git,b aber sich das auf eine contrib- zurückführen lässt
* study "component - error" correlation
    * by eye?
    * plot correlations?


    
DIAGNOSE 1


* is "e_3d" correct, but there should be another term to even it out?

* near field > Lt(Real+Images) > e_3d (e_elc=0) > _run_elc_on_system
    * seems like only the e_3d components contribute (e_elc=0) - but thats based on p3m ???
        * do i use p3m wrong?
        * is p3m wrong?
"""
