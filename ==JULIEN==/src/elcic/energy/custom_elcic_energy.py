import espressomd
import espressomd.electrostatics
import numpy as np


def get_elcic_energy_contribs(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs = parts.q.copy()
    pos = parts.pos.copy()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    """
    can only have one instance of "system" at once

        system.part.clear() # remove all particles
        system.box_l = orig_box # can only reset box size when no particles are present
        system.part.add(pos=pos, q=q)
    
    use p3m as such:
        p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=pw_error,
        check_neutrality=False,
        verbose=False,
        )
        system.electrostatics.solver = p3m
        system.integrator.run(0)
        e_3d = float(system.analysis.energy()["total"])

    """
    

   
    contribs = {
        "e_near": float(e_near),
        "e_far":  float(e_far_total),
        # Sub-breakdowns for diagnostics
        "l0": {
            "e_3d":   e_L0_3d,
            "e_elc":  e_L0_elc,
            "total":  e_L0_total,
        },
        "pm1": {
            "e_3d":   e_L1_3d,
            "e_elc":  e_L1_elc,
            "total":  e_L1_total,
        },
        "lt": {
            "e_3d":   e_LT_3d,
            "e_elc":  e_LT_elc,
            "total":  e_LT_total,
        },
        "e_far_detail": e_far_detail,
    }
    return contribs


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    
    contribs = get_elcic_energy_contribs(
        system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
    )
    return contribs["e_near"] + contribs["e_far"]