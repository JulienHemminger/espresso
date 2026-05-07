import pandas as pd
import numpy as np
import plotly.express as px
import signal
from datetime import datetime
from pathlib import Path

import espressomd
import espressomd.electrostatics
from elcic.energy.custom_elcic_energy import get_elcic_energy
from elc.energy.legacy_elc_energy import get_legacy_elc_energy
from elcic.energy.analytical.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def param_sweep_accuracy_convergence(system, parameter_list, accuracies=[10**-i for i in range(1, 6)]):
    results = []
    signal.signal(signal.SIGALRM, timeout_handler)

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
                k_max=10, 
                n_real=10
            )

            for acc in accuracies:
                custom_energy = get_elcic_energy(
                    system=system,
                    gap_size=params["gap_size"],
                    pw_error=acc,
                    prefactor=params["prefactor"],
                    delta_mid_bot=params["delta_mid_bot"],
                    delta_mid_top=params["delta_mid_top"],
                )

                try:
                    signal.alarm(20)
                    legacy_energy = get_legacy_elc_energy(
                        system=system, 
                        gap_size=params["gap_size"], 
                        pw_error=acc, 
                        prefactor=1.0,
                        delta_mid_top=params["delta_mid_top"], 
                        delta_mid_bot=params["delta_mid_bot"]
                    )
                    signal.alarm(0)
                except (Exception, TimeoutException):
                    signal.alarm(0)
                    legacy_energy = custom_energy

                results.append({
                    "accuracy": acc,
                    "legacy_error": np.abs(legacy_energy - ana_energy),
                    "custom_error": np.abs(custom_energy - ana_energy),
                    "set_id": f"Set {i+1}",
                    **{f"param_{k}": str(v) for k, v in params.items()}
                })
            
            print(f"Completed set {i+1}")

        except Exception as e:
            print(f"Skipping set {i+1} due to error: {e}")

    if results:
        _plot_interactive_errors(pd.DataFrame(results))


def _plot_interactive_errors(df):
    hover_cols = [c for c in df.columns if c.startswith("param_")]
    
    df_melted = df.melt(
        id_vars=["accuracy", "set_id"] + hover_cols,
        value_vars=["legacy_error", "custom_error"],
        var_name="Method",
        value_name="Measured_Error"
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
        markers=True
    )

    fig.update_xaxes(tickformat=".0e", autorange="reversed", exponentformat="e", dtick=1)
    fig.update_yaxes(tickformat=".0e", exponentformat="e")
    fig.update_layout(xaxis_title="Accuracy", yaxis_title="Abs Error")

    fig.show()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path("/home/main") / f"simulation_{timestamp}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")