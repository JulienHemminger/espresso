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
    Computes both Legacy and Custom ELCIC errors.
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

            # Analytical energy calculation (Reference)
            ana_energy = analytical_single_plate_2d_ewald_elcic_energy(
                params["positions"], params["charges"], system.box_l, 
                params["prefactor"], params["delta_mid_bot"], k_max=10, n_real=10
            )

            for acc in accuracies:
                # 1. Legacy Error
                legacy_energy = get_legacy_elc_energy(
                    system, params["gap_size"], acc, 
                    params["delta_mid_top"], params["delta_mid_bot"]
                )
                legacy_err = np.abs(legacy_energy - ana_energy)

                # 2. Custom Error
                contribs = get_elcic_energy_contribs(
                    system,
                    params["gap_size"],
                    acc,
                    params["prefactor"],
                    params["delta_mid_bot"],
                    params["delta_mid_top"],
                )
                custom_energy = contribs["e_far"] + contribs["e_near"]
                custom_err = np.abs(custom_energy - ana_energy)
                
                entry = {
                    "accuracy": acc,
                    "legacy_error": legacy_err,
                    "custom_error": custom_err,
                    "set_id": f"Set {p_idx+1}",
                    **{f"param_{k}": str(v) for k, v in params.items()}
                }
                results.append(entry)
            
            print(f"Successfully completed set {p_idx+1}")

        except Exception as e:
            print(f"Error in parameter set {p_idx+1}: {e}. Skipping...")
            continue
                
    return pd.DataFrame(results)

def plot_interactive_errors(df):
    if df.empty:
        print("No data to plot.")
        return

    # Identify parameter columns for hover data
    hover_cols = [c for c in df.columns if c.startswith("param_")]
    
    # Reshape dataframe for Plotly Express (Long format)
    # This creates a 'Method' column with values 'legacy_error' or 'custom_error'
    df_melted = df.melt(
        id_vars=["accuracy", "set_id"] + hover_cols,
        value_vars=["legacy_error", "custom_error"],
        var_name="Method",
        value_name="Measured_Error"
    )

    # Plot using line_dash to distinguish between error types
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
        title="ELCIC Error Convergence: Legacy (Dotted) vs Custom (Solid)",
        markers=True
    )

    # Scientific notation and axis reversal logic
    fig.update_xaxes(
        tickformat=".0e", 
        autorange="reversed",
        exponentformat="e",
        dtick=1
    )
    
    fig.update_yaxes(
        tickformat=".0e",
        exponentformat="e"
    )

    fig.update_layout(
        xaxis_title="Requested Accuracy (pw_error)",
        yaxis_title="Energy Error (Absolute)",
        hovermode="closest",
        legend_title_text="Parameter Set / Method"
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
            "delta_mid_bot": -1.0,
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