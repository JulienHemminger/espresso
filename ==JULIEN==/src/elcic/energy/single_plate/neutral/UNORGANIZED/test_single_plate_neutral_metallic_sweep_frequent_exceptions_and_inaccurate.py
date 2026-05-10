import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path

import espressomd
from elcic.energy.custom_elcic_energy import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy
from elcic.energy.single_plate.neutral.UNORGANIZED.analytical_single_plate_elcic_energy import (
    analytical_single_plate_2d_ewald_elcic_energy,
)
pw_error = 1e-8
accuracy_max = 10


def param_sweep_accuracy_convergence(
    system, parameter_list, accuracies=np.logspace(-1, -accuracy_max, num=10)
):
    results = []

    for i, params in enumerate(parameter_list):
        try:
            system.part.clear()
            system.box_l = [params["lx"], params["ly"], params["lz"]]
            for pos, q in zip(params["positions"], params["charges"]):
                system.part.add(pos=pos, q=q)

            ana_energy = analytical_single_plate_2d_ewald_elcic_energy(
                positions=params["positions"],
                charges=params["charges"],
                box_l=system.box_l,
                prefactor=params["prefactor"],
                delta_mid_bot=params["delta_mid_bot"],
                accuracy=pw_error
            )

            for acc in accuracies:
                print("Computing custom_energy...")
                custom_energy = get_elcic_energy(
                    system=system,
                    gap_size=params["gap_size"],
                    pw_error=acc,
                    prefactor=params["prefactor"],
                    delta_mid_bot=params["delta_mid_bot"],
                    delta_mid_top=params["delta_mid_top"],
                )
                print("Computing legacy_energy...")
                legacy_energy = get_legacy_elc_energy(
                    system=system,
                    gap_size=params["gap_size"],
                    pw_error=acc,
                    prefactor=params["prefactor"],
                    delta_mid_top=params["delta_mid_top"],
                    delta_mid_bot=params["delta_mid_bot"],
                )

                results.append(
                    {
                        "accuracy": acc,
                        "legacy_error": np.abs(legacy_energy - ana_energy),
                        "custom_error": np.abs(custom_energy - ana_energy),
                        "set_id": f"Set {i + 1}",
                        **{f"param_{k}": str(v) for k, v in params.items()},
                    }
                )

            print(f"Completed set {i + 1}")

        except Exception as e:
            print(f"Skipping set {i + 1} due to error: {e}")

    if results:
        _plot_interactive_errors(pd.DataFrame(results))

def _plot_interactive_errors(df):
    hover_cols = [c for c in df.columns if c.startswith("param_")]

    df_melted = df.melt(
        id_vars=["accuracy", "set_id"] + hover_cols,
        value_vars=["legacy_error", "custom_error"],
        var_name="Method",
        value_name="Measured_Error",
    )

    fig = px.line(
        df_melted,
        x="accuracy",
        y="Measured_Error",
        color="set_id",
        line_dash="Method",
        line_dash_map={"legacy_error": "dot", "custom_error": "solid"},
        log_x=True,
        log_y=True,
        hover_data=hover_cols,
        title="ELCIC Error Convergence",
        markers=True,
    )

    fig.update_xaxes(
        tickformat=".0e", autorange="reversed", exponentformat="e", dtick=1
    )
    fig.update_yaxes(tickformat=".0e", exponentformat="e")
    fig.update_layout(xaxis_title="Accuracy", yaxis_title="Abs Error")

    fig.show()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path("/home/main") / f"simulation_{timestamp}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")





system = espressomd.System(box_l=[1, 1, 1])
system.time_step = 0.01

params_count = 1
params_sets = []

for i in range(params_count):
    params = {
        "lx": np.random.uniform(10.0, 50.0),
        "ly": np.random.uniform(10.0, 50.0),
        "lz": np.random.uniform(10.0, 40.0),
        "gap_size": np.random.uniform(10.0, 20.0),
        "prefactor": 1.0,
        "delta_mid_top": 0.0,
        "delta_mid_bot": -1.0,
        "pw_error": pw_error,
        "charges": [+1, -1],
    }

    eps = 1e-1
    params["positions"] = [
        np.array(
            [
                np.random.uniform(eps, params["lx"] - eps),
                np.random.uniform(eps, params["ly"] - eps),
                np.random.uniform(eps, params["lz"] - params["gap_size"] - eps),
            ]
        )
        for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)


param_sweep_accuracy_convergence(system, params_sets)
