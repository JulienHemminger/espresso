import numpy as np
import espressomd
import espressomd.electrostatics


def _run_p3m_energy(system, prefactor, accuracy):
    p3m = espressomd.electrostatics.P3M(
        prefactor=prefactor,
        accuracy=accuracy,
        check_neutrality=False,
        verbose=False,
    )
    system.electrostatics.solver = p3m
    system.integrator.run(0)
    return system.analysis.energy()["total"]


def _dipole_non_neutral_term(box, qs, zs, prefactor):
    lx, ly, lz = box
    shift = lz / 2.0

    xi0 = np.sum(qs)
    xi1 = np.sum(qs * (zs - shift))
    xi2 = np.sum(qs * (zs - shift) ** 2)

    pref = prefactor * 2.0 * np.pi / (lx * ly * lz)

    energy = 2.0 * pref * xi1**2
    energy += 2.0 * pref * (-xi0 * xi2 - (lz**2 / 12.0) * xi0**2)

    return energy


def get_elcic_energy(system, params):

    lx = params["lx"]
    ly = params["ly"]
    lz = params["lz"]

    gap = params["gap_size"]
    prefactor = params["prefactor"]
    accuracy = params["pw_error"]
    delta_bot = params["delta_mid_bot"]
    delta_top = params["delta_mid_top"]

    qs = np.array(params["charges"])
    ps = np.array(params["positions"])

    # box height without gap
    box_h = lz - gap

    # -----------------------------------
    # Save original system
    # -----------------------------------
    system.part.clear()
    system.box_l = [lx, ly, lz]
    system.part.add(pos=ps, q=qs)

    # -----------------------------------
    # 1) Real charges only
    # -----------------------------------
    e_real = _run_p3m_energy(system, prefactor, accuracy)

    # -----------------------------------
    # 2) Real + image charges
    # -----------------------------------
    ps_img = []
    qs_img = []

    for p, q in zip(ps, qs):
        z = p[2]

        if z < gap:
            ps_img.append([p[0], p[1], -z])
            qs_img.append(delta_bot * q)

        if z > box_h:
            ps_img.append([p[0], p[1], 2 * box_h - z])
            qs_img.append(delta_top * q)

    if len(ps_img) > 0:
        ps_tot = np.vstack([ps, ps_img])
        qs_tot = np.concatenate([qs, qs_img])
    else:
        ps_tot = ps.copy()
        qs_tot = qs.copy()

    system.part.clear()
    system.part.add(pos=ps_tot, q=qs_tot)

    e_real_plus_img = _run_p3m_energy(system, prefactor, accuracy)

    # -----------------------------------
    # 3) Image charges only
    # -----------------------------------
    if len(ps_img) > 0:
        system.part.clear()
        system.part.add(pos=np.array(ps_img), q=np.array(qs_img))
        e_img = _run_p3m_energy(system, prefactor, accuracy)
    else:
        e_img = 0.0

    # -----------------------------------
    # Restore original
    # -----------------------------------
    system.part.clear()
    system.part.add(pos=ps, q=qs)

    # -----------------------------------
    # Assemble exactly like C++
    # -----------------------------------
    energy = (
        0.5 * e_real
        + 0.5 * e_real_plus_img
        - 0.5 * e_img
    )

    # Add Yeh–Berkowitz dipole correction
    energy += _dipole_non_neutral_term([lx, ly, lz], qs, ps[:, 2], prefactor)

    return energy