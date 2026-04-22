import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import espressomd
import espressomd.electrostatics
from elcic.energy.custom_elcic_energy import get_elcic_energy_contribs
from elc.energy.legacy_elc_energy import get_legacy_elc_energy
from elcic.energy.analytical.analytical_single_plate_elcic_energy import analytical_single_plate_2d_ewald_elcic_energy

def run_multiple_parameter_sets(system, parameter_list):
    """
    Runs simulations with error handling. Skips failed parameter sets.
    """
    results = []
    accuracies = [10**-i for i in range(1, 10)]

    for p_idx, params in enumerate(parameter_list):
        try:
            # Update system/particles
            system.part.clear()
            system.box_l = [params["lx"], params["ly"], params["lz"]]
            for pos, q in zip(params["positions"], params["charges"]):
                system.part.add(pos=pos, q=q)

            # Analytical energy calculation
            ana_energy = analytical_single_plate_2d_ewald_elcic_energy(
                params["positions"], params["charges"], system.box_l, 
                params["prefactor"], params["delta_mid_bot"], k_max=10, n_real=10
            )

            for acc in accuracies:
                legacy_energy = get_legacy_elc_energy(
                    system, params["gap_size"], acc, 
                    params["delta_mid_top"], params["delta_mid_bot"]
                )
                
                error = np.abs(legacy_energy - ana_energy)
                
                entry = {
                    "accuracy": acc,
                    "legacy_error": error,
                    "set_id": f"Set {p_idx+1}",
                    **{f"param_{k}": str(v) for k, v in params.items()}
                }
                results.append(entry)
            
            print(f"Successfully completed set {p_idx+1}")

        except Exception as e:
            # Catch errors (e.g., P3M tuning failures, invalid geometries)
            print(f"Error in parameter set {p_idx+1}: {e}. Skipping...")
            continue
                
    return pd.DataFrame(results)

def plot_interactive_errors(df):
    if df.empty:
        print("No data to plot.")
        return

    hover_cols = [c for c in df.columns if c.startswith("param_")]

    fig = px.line(
        df, 
        x="accuracy", 
        y="legacy_error", 
        color="set_id",
        log_x=True, 
        log_y=True,
        hover_data=hover_cols,
        title="Legacy Error Convergence (Scientific Notation)",
        markers=True
    )

    # Use '.0e' for standard scientific notation (e.g., 1e-6)
    # The tickformat ".1e" would give 1.0e-6
    fig.update_xaxes(
        tickformat=".0e", 
        autorange="reversed",
        exponentformat="e", # Forces the 'e' notation specifically
        dtick=1 # Ensures a tick for every order of magnitude
    )
    
    fig.update_yaxes(
        tickformat=".0e",
        exponentformat="e"
    )

    fig.update_layout(
        xaxis_title="Requested Accuracy (pw_error)",
        yaxis_title="Measured Legacy Error",
        hovermode="closest"
    )
    
    fig.show()

    filename = "simulation_results.html"
    fig.write_html(filename)
    print(f"Interactive plot saved to {filename}")



def generate_random_param_sets(n_sets=3):
    param_sets = []
    for _ in range(n_sets):
        p = {
            "lx": np.random.uniform(8.0, 15.0),
            "ly": np.random.uniform(8.0, 15.0),
            "lz": np.random.uniform(10.0, 20.0),
            "gap_size": np.random.uniform(5.0, 10.0),
            "prefactor": 1.0,
            "delta_mid_top": 0.0,
            "delta_mid_bot": -1.0,  # Keeping metallic constraint
            "charges": [+1, -1],
            "pw_error": 1e-8,
            "positions": [
                np.array([np.random.uniform(0.1, 7.0), np.random.uniform(0.1, 7.0), 0.01]),
                np.array([np.random.uniform(0.1, 7.0), np.random.uniform(0.1, 7.0), 0.02])
            ]
        }
        param_sets.append(p)
    return param_sets

# Execution
if __name__ == "__main__":
    param_sets = generate_random_param_sets(n_sets=10)
    system = espressomd.System(box_l=[10, 10, 10])
    system.time_step = 0.01
    
    df = run_multiple_parameter_sets(system, param_sets)
    plot_interactive_errors(df)