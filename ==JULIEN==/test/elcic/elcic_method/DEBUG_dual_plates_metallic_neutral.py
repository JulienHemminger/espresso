import matplotlib.pyplot as plt
import numpy as np
import time
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs
import numpy as np
import espressomd
import numpy as np

from test.elcic.elcic_method.common.elcic_energy_accuracy_convergence import run as run_elcic_energy_accuracy_convergence
from test.elcic.elcic_method.common.elcic_dipole_shifting import run as run_elcic_dipole_shifting

import matplotlib.pyplot as plt
import numpy as np
import time
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs

import matplotlib.pyplot as plt
import numpy as np
import time
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs
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
    
    # Range as requested: from eps to (lz - gap - eps)
    # Note: If you want to check full symmetry, you may need to sweep 
    # beyond (lz - gap) if the gap is large.
    z_range = np.linspace(eps, lz - gap - eps, num=z_pos_count)
    
    data = {
        "z": [],
        "custom_total": [],
        "legacy_total": [],
        "e_near": [],
        "e_far": [],
        "l0": [],
        "pm1": [],
        "lt": []
    }

    for z in z_range:
        system.part.clear()
        # Maintain original x,y while sweeping z
        for i in range(len(params["charges"])):
            orig_pos = params["positions"][i]
            system.part.add(pos=[orig_pos[0], orig_pos[1], z], q=params["charges"][i])

        # 1. Custom ELCIC Breakdown
        contribs = get_elcic_energy_contribs(
            system, gap, pw_error, prefactor, delta_mid_bot, delta_mid_top
        )
        
        # 2. Legacy ELC (Total only)
        # Re-adding particles isn't strictly necessary if get_legacy handles current system state
        e_legacy = get_legacy_elc_energy(system, gap, prefactor, pw_error, delta_mid_top, delta_mid_bot)

        data["z"].append(z)
        data["custom_total"].append(contribs["e_near"] + contribs["e_far"])
        data["legacy_total"].append(e_legacy)
        data["e_near"].append(contribs["e_near"])
        data["e_far"].append(contribs["e_far"])
        data["l0"].append(contribs["l0"]["total"])
        data["pm1"].append(contribs["pm1"]["total"])
        data["lt"].append(contribs["lt"]["total"])

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    z_vals = np.array(data["z"])

    # Top Plot: High-level Comparison
    axes[0].plot(z_vals, data["custom_total"], 'c-', lw=2, label='Total Custom ELCIC')
    axes[0].plot(z_vals, data["legacy_total"], 'm-', lw=2, label='Total Legacy ELCIC')
    axes[0].plot(z_vals, data["e_near"], 'r:', label='Custom: Near Field')
    axes[0].plot(z_vals, data["e_far"], 'g:', label='Custom: Far Field')
    axes[0].set_title("Energy Comparison: Custom vs Legacy")
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].grid(True)

    # Middle Plot: Custom ELCIC Internal Near-Field Components
    axes[1].plot(z_vals, data["l0"], 'r-', label='L0 (Real)')
    axes[1].plot(z_vals, data["pm1"], 'g--', label='L-1 + L+1 (Images)')
    axes[1].plot(z_vals, data["lt"], 'b:', label='Lt (Real + Images)')
    axes[1].set_title("Custom ELCIC Near-Field (P3M) Internal Sets")
    axes[1].set_ylabel("Energy")
    axes[1].legend()
    axes[1].grid(True)

    # Bottom Plot: Symmetry Check
    # Residual = E(z) - E(flipped_z)
    custom_res = np.array(data["custom_total"]) - np.flip(data["custom_total"])
    legacy_res = np.array(data["legacy_total"]) - np.flip(data["legacy_total"])
    
    axes[2].plot(z_vals, custom_res, 'c--', label='Custom Symmetry Res.')
    axes[2].plot(z_vals, legacy_res, 'm--', label='Legacy Symmetry Res.')
    axes[2].axhline(0, color='red', alpha=0.3)
    axes[2].set_title("Symmetry Check: E(z) - E(Mirror_z)")
    axes[2].set_xlabel("z-position")
    axes[2].set_ylabel("Difference")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
# To use:

system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params = {
    "lx": 50.0,
    "ly": 50.0,
    "lz": 11.0,
    "gap_size": 7.0,
    "prefactor": 1.0,
    "delta_mid_top": 0.3,
    "delta_mid_bot": 0.3, # for metallic it must: Δ = −1
    "charges": [+1, -1],
    'pw_error': 1e-6,
    "positions": [np.array([1, 2, 3]), np.array([4, 5, 1])],
    "title": "Dual Plates, Both Metallic, Neutral"
}
run_elcic_diagnostic(system, z_pos_count=4, params=params)