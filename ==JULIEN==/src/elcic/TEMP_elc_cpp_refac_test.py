from elc.energy.legacy_elc_energy import get_legacy_energy
import numpy as np
import espressomd
import math


NESSECARY_KEYS = ["lx", "ly", "lz", "gap_size", "pw_error", "prefactor"]
OPTIONAL_KEYS = ["delta_mid_top", "delta_mid_bot"]


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return (1 - t) * a + t * b


def lerp_dict(start_params, end_params, t):
    """Modular interpolation of parameters."""
    lerp_params = {
        key: lerp(start_params[key], end_params[key], t) for key in NESSECARY_KEYS
    }

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


start_params = {
    "lx": 10.0,
    "ly": 10.0,
    "gap_size": 10.0,
    "prefactor": 1.0,
    "delta_mid_top": -1.0,
    "delta_mid_bot": -1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 4]),
        np.array([3, 2, 4]),
    ],  # legacy fails for part.z <= 3
}
start_params["lz"] = start_params["gap_size"] + 10


end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "gap_size": 15.0,  # legacy runs with: 14, 15, fails with 16, 20
    "prefactor": 1.0,
    "delta_mid_top": +1.0,
    "delta_mid_bot": +1.0,
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([6, 5, 30]),
        np.array([3, 2, 30]),
    ],  # legacy runs for part.z = 4, 24, fails for part.z = 3, 34, 39
}
end_params["lz"] = start_params["gap_size"] + 40

system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = (
    0.4  # required to fix "tuning failed: number of cells 6 is smaller than minimum 8"
)

N = 10
energies = []
for t in np.linspace(0, 1, num=N):
    params = lerp_dict(start_params, end_params, t)

    system.electrostatics.clear()
    system.part.clear()
    system.box_l = [params["lx"], params["ly"], params["lz"]]
    for i in range(len(params["charges"])):
        system.part.add(pos=params["positions"][i], q=params["charges"][i])

    energies.append(get_legacy_energy(system, params))

print(f"{N=}: {energies=}")

ABS_TOL = 9e-7 # elc.cpp isnt deterministic. for same params, it yields results within 9e-7
ground_truth_energies_N10=[-0.2765154822643574, -0.2500702524722067, -0.24313750776329318, -0.24012013430195772, -0.2380948418255439, -0.2363104533087212, -0.23452703427924, -0.23264505761174437, -0.23060694727871123, -0.2283717076533031]

assert len(energies) == len(ground_truth_energies_N10)
assert all(
    math.isclose(a, b, abs_tol=ABS_TOL) for a, b in zip(energies, ground_truth_energies_N10)
)





"""
Action Tree
* find the entry point when i call espresso.ELC in python

* print contribs and params on C++ side

* one by one, recreate contribs in python unil i have working custom.py


* do the same with forces

"""