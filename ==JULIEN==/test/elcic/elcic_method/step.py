import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np

def run_simulation(system, lx, ly, lz, gap_size, charges, positions, pw_error, z_pos_count, params, **kwargs):
    system.box_l = [lx, ly, lz]
    
    # Initialize solver once
    p3m = espressomd.electrostatics.P3M(prefactor=1.0, accuracy=pw_error)
    elc = espressomd.electrostatics.ELC(actor=p3m, gap_size=gap_size, maxPWerror=pw_error, 
                                        check_neutrality=False, **kwargs)
    eps = 0.5
    z_range = np.linspace(eps, lz - gap_size - eps, num=z_pos_count)
    energies = []

    for z in z_range:
        system.part.clear()
        for q, pos in zip(charges, positions):
            system.part.add(pos=[pos[0], pos[1], z], q=q)
        
        system.integrator.run(0)
        system.electrostatics.solver = elc
        energies.append(system.analysis.energy()["total"])

    # Plotting
    plt.plot(z_range, energies, 'x--', label='ELC Energy')
    text_str = "\n".join(f"{k}: {v}" for k, v in params.items() if isinstance(v, (int, float)))
    plt.gca().text(0.95, 0.95, text_str, transform=plt.gca().transAxes, va='top', ha='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), fontsize=9)
    plt.xlabel('z-position'); plt.ylabel('Energy'); plt.grid(True); plt.legend()
    plt.show()

# Setup
system = espressomd.System(box_l=[1, 1, 1])
system.time_step, system.cell_system.skin = 0.01, 0.4
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

run_simulation(system, **params, z_pos_count=16, params=params)