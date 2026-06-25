from common.plotting.param_lerp_plot import run_lerp_plot
import numpy as np
import espressomd
from elc.energy._5_param_sweep_rdm_tests.custom_elc_energy_for_accuracy_convergence import (
    get_elc_energy,
)
from elc.energy._1_big_box_neutral_dipole.reference_solution.get_direct_sum_energy import (
    get_direct_sum_energy as get_direct_sum_energy,
)


def run_madelung(system, ions_list, gap_size=1, accuracy=1e-6):
    madelung_refs = []
    elc_energies = []

    for ions_per_axis in ions_list:
        system.part.clear()
        system.box_l = [10, 10, 10]

        l_xy = min(system.box_l[0], system.box_l[1])

        spacing = l_xy / ions_per_axis
        ion_z_pos = system.box_l[2] / 2.0
        for i in range(ions_per_axis):
            for j in range(ions_per_axis):
                charge = (-1.0) ** (i + j)
                system.part.add(pos=[i * spacing, j * spacing, ion_z_pos], q=charge)

        ion_count = len(system.part)
        MADELUNG_CONSTANT_PERFECT_2D_SHEET = (
            1.6155426267128247  # e.g. perfect 2D sheet of NaCl
        )
        madelung_2d_ref = (
            -MADELUNG_CONSTANT_PERFECT_2D_SHEET * ion_count / (2.0 * spacing)
        )
        madelung_refs.append(madelung_2d_ref)

        elc_energy = get_elc_energy(system, gap_size, accuracy)
        elc_energies.append(elc_energy)

        print(f"{ions_per_axis=}: diff={np.abs(madelung_2d_ref - elc_energy)}")

    return madelung_refs, elc_energies


# --- 1. Define custom evaluation wrapper functions ---
# These functions will automatically receive (system, params) from the generalized script


def evaluate_analytical(system, params):
    """Computes reference energy based on the nearest even ion count."""
    # Convert the continuous lerp float into the closest even integer
    ions = int(2 * round(params["ions_per_axis"] / 2))

    # Run your madelung logic for just this ion count
    refs, _ = run_madelung(system, [ions])
    return refs[0]


def evaluate_elc(system, params):
    """Computes ELC energy based on the nearest even ion count."""
    ions = int(2 * round(params["ions_per_axis"] / 2))

    # Run your madelung logic for just this ion count
    _, elcs = run_madelung(system, [ions])
    return elcs[0]


# --- 2. Setup the Espressomd System ---
system = espressomd.System(box_l=[50, 50, 50])
system.time_step = 0.01
system.cell_system.skin = 0.4


# --- 3. Define Parameters for the Lerp Engine ---
# Standard required structural parameters are held constant,
# while 'ions_per_axis' spans across your range.
start_params = {
    "lx": 50.0,
    "ly": 50.0,
    "lz": 50.0,
    "positions": [],
    "charges": [],  # Handled internally by your run_madelung function
    "ions_per_axis": 2.0,
}

end_params = {
    "lx": 50.0,
    "ly": 50.0,
    "lz": 50.0,
    "positions": [],
    "charges": [],
    "ions_per_axis": 20.0,
}

# 10 steps spanning from 2 to 20 precisely hits: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
steps = 10


# --- 4. Map Metrics with Hashable Tuples ---
# Since you requested to plot only absolute values (no error plot)
metrics_to_plot = {
    (
        "Analytical Reference",
        (("color", "red"), ("marker", "o"), ("linestyle", "None"), ("alpha", 0.7)),
    ): evaluate_analytical,
    (
        "ELC Energy",
        (("color", "tab:blue"), ("marker", "x"), ("linestyle", "-"), ("alpha", 0.7)),
    ): evaluate_elc,
}


# --- 5. Execute ---
run_lerp_plot(
    system=system,
    start_params=start_params,
    end_params=end_params,
    lerp_step_count=steps,
    plot_metrics=metrics_to_plot,
)
