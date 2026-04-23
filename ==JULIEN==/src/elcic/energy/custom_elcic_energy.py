import espressomd
import espressomd.electrostatics
import numpy as np


def get_elcic_energy(
    system, gap_size, pw_error, prefactor, delta_mid_bot, delta_mid_top
):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor, accuracy=pw_error, check_neutrality=False, verbose=False
    )

    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T

    # 1. 3D Periodic Energy from P3M
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    e_3d = system.analysis.energy()["total"]

    """
    Note: You can only have one instance of the system class at a time.
    system.part.clear() to remove all particles
    system.part.add(pos=pos, q=q) to add a particle


    system.box_l = [..] to set the system size (all particles have to be removed before this)
    """

    pass