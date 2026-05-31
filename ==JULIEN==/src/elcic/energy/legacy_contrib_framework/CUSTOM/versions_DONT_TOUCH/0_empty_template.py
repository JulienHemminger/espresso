import espressomd
import espressomd.electrostatics
import numpy as np



def get_elcic_energy(system, params: dict):
    """
    Computes electrostatic energy for 2D+h system with TOP/BOTTOM dielectric interfaces using ELCIC.
    """
    box = np.array(system.box_l)
    lz_full = box[2]
    gap, eps = params["gap_size"], params["pw_error"]
    pref = params.get("prefactor", 1.0)
    db, dt = params["delta_mid_bot"], params["delta_mid_top"]
    lz = lz_full - gap
    parts = system.part.all()
    charges, positions = parts.q.copy(), parts.pos.copy()
    lambda_ = np.clip(params.get("lambda", lz / 2), 1e-3, lz / 2)


    e_near_3d = 0
    e_near_corr = 0

    e_near = e_near_3d + e_near_corr
    e_far = 0
    e_total = e_near + e_far

    return {
        "E_total": e_total,
        "E_near": e_near,
        "E_near_p3m": e_near_3d,
        "E_near_corr": e_near_corr,
        "E_far": e_far,
    }
"""
example_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([6, 5, 4]), np.array([3, 2, 4])],
}
example_params["lz"] = example_params["gap_size"] + 10




espressomd.System is a singleton, you can only have one instance

to initialize a system:
    system = espressomd.System(box_l=[1, 2, 3])
    system.time_step = 0.01

 system.part.clear() # remove all particles

 system.part.add(pos=np.array([1, 2, 3]), q=1) # to add a particle

 system.box_l = [lx, ly, lz] # to resize a system, system needs to have no particles before doing this.

to use P3M:
    p3m = espressomd.electrostatics.P3M(
            prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
        )
    system.electrostatics.solver = p3m
    system.integrator.run(0)#
    energy = system.analysis.energy()["total"]
"""