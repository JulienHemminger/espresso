import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Configuration for easier maintenance
NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor"]
OPTIONAL_KEYS = ["delta_mid_top", "delta_mid_bot"]

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b

def lerp_dict(start_params, end_params, t):
    """Modular interpolation of parameters."""
    lerp_params = {key: lerp(start_params[key], end_params[key], t) for key in NESSECARY_KEYS}
    
    for key in OPTIONAL_KEYS:
        if key in start_params and key in end_params:
            lerp_params[key] = lerp(start_params[key], end_params[key], t)
            
    lerp_params["positions"] = [
        lerp(np.array(p_start), np.array(p_end), t)
        for p_start, p_end in zip(start_params["positions"], end_params["positions"])
    ]
    lerp_params["charges"] = [
        lerp(q_start, q_end, t)
        for q_start, q_end in zip(start_params["charges"], end_params["charges"])
    ]
    return lerp_params

def run_lerp_plot(system, start_params, end_params, get_custom_energy, get_analytical_energy=None, get_legacy_energy=None, steps=20):
    t_values = np.linspace(0, 1, steps)
    results = {"legacy": [], "analytical": [], "custom": []}

    # Run Simulation/Evaluation Loop
    for t in t_values:
        system.electrostatics.clear()
        params = lerp_dict(start_params, end_params, t)

        system.part.clear()
        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])

        if get_legacy_energy: results["legacy"].append(get_legacy_energy(system, params))
        if get_analytical_energy: results["analytical"].append(get_analytical_energy(params))
        results["custom"].append(get_custom_energy(system, params))

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 1. Top Subplot: Absolute Energy
    ax1.plot(t_values, results["custom"], label="Custom Energy", color="green", lw=2)
    if get_analytical_energy: ax1.plot(t_values, results["analytical"], label="Analytical Energy", color="black", ls="--")
    if get_legacy_energy: ax1.plot(t_values, results["legacy"], label="Legacy Energy", color="orange", ls=":")
    ax1.set_ylabel("Total Energy")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # 2. Bottom Subplot: Error
    if get_analytical_energy:
        ax2.plot(t_values, np.abs(np.array(results["custom"]) - np.array(results["analytical"])), label="Error: Custom-Ana", color="blue")
        if get_legacy_energy: ax2.plot(t_values, np.abs(np.array(results["legacy"]) - np.array(results["analytical"])), label="Error: Leg-Ana", color="red")
        ax2.set_ylabel("Absolute Energy Error")
    elif get_legacy_energy:
        ax2.plot(t_values, np.abs(np.array(results["custom"]) - np.array(results["legacy"])), label="Diff: Custom-Leg", color="purple")
        ax2.set_ylabel("Difference")
    
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Interpolation Parameter $t$")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # Verbose Label Construction
    label_lines = ["**Parameters**"]
    for key in start_params.keys():
        v1, v2 = start_params[key], end_params[key]
        if key == "positions":
            label_lines.append("positions:")
            for i, (p1, p2) in enumerate(zip(v1, v2)):
                label_lines.append(f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1)}] → [{', '.join(f'{x:.1f}' for x in p2)}]")
        else:
            label_lines.append(f"{key}: {v1} → {v2}")

    fig.text(0.85, 0.5, "\n".join(label_lines), verticalalignment='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), transform=fig.transFigure)

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()