import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np


def get_legacy_elc_energy(
    system, gap_size, pw_error, delta_mid_top=None, delta_mid_bot=None
):
    p3m = espressomd.electrostatics.P3M(
        prefactor=1.0, accuracy=pw_error
    )

    args = {
        "actor": p3m,
        "gap_size": gap_size,
        "maxPWerror": pw_error,
        "check_neutrality": False,
        "neutralize": False,
    }

    args["delta_mid_top"] = delta_mid_top
    args["delta_mid_bot"] = delta_mid_bot

    elc_legacy = espressomd.electrostatics.ELC(**args)

    system.electrostatics.solver = elc_legacy
    system.integrator.run(0)
    return system.analysis.energy()["total"]


def run(system, lx, ly, lz, gap_size, charges, positions, prefactor, pw_error, delta_mid_top, delta_mid_bot, z_pos_count, params):
    eps = 0.5
    system.part.clear()
    system.box_l = [lx, ly, lz]

    legacy_energies = []

    z_range = np.linspace(eps, lz - gap_size - eps, num=z_pos_count)
    for z in z_range:
        system.part.clear()
        for i in range(min(len(charges), len(positions))):
            # TREPPE
            pos = positions[i]
            pos[2] = z
            system.part.add(pos=pos, q=charges[i])   
            
            # GLATT
            pos = positions[i]
            system.part.add(pos=[pos[0], pos[1], z], q=charges[i])
            
        legacy_energies.append(get_legacy_elc_energy(system, gap_size, pw_error, delta_mid_top, delta_mid_bot))  
       

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, legacy_energies, label='Legacy ELC', marker='x', linestyle='--')

    plt.text(0.95, 0.95, "\n".join([f"{k}: {v}" for k, v in params.items()]), transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
             fontsize=9, family='monospace')
    
    plt.xlabel('z-position')
    plt.ylabel('Energy')
    plt.title('Energy Comparison vs Particle Position')
    plt.legend()
    plt.grid(True)
    plt.show()

   



system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01
system.cell_system.skin = 0.4
z = 1
params = {
        "lx": 9.0,
        "ly": 12.0,
        "lz": 19.0,
        "gap_size": 15.0,
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -0.6,
        "charges": [+1, -1],
        'pw_error': 1e-8,
        "positions": [np.array([6, 5, z]), np.array([1, 3, z])]
    }

run(system, **params, z_pos_count=16, params=params)
