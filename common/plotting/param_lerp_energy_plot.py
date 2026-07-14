import matplotlib.pyplot as plt
import numpy as np

from common.plotting.utils.param_lerping import lerp_dict
from common.plotting.utils.plot_saving import save_plot_with_timestamp


def validate_inputs(start_params: dict, end_params: dict, lerp_step_count: int):
    """Assertions to ensure matching parameter dictionary structures and valid step count."""
    assert isinstance(start_params, dict) and isinstance(end_params, dict), (
        "Parameters must be dictionaries."
    )
    assert set(start_params.keys()) == set(end_params.keys()), (
        "Start and end parameter keys do not match."
    )
    assert lerp_step_count > 1, "lerp_step_count must be greater than 1."

    # Check that required core structural geometries are present
    for core_key in ["lx", "ly", "lz", "positions", "charges"]:
        assert core_key in start_params, (
            f"Missing critical system parameter: '{core_key}'"
        )

    assert len(start_params["positions"]) == len(start_params["charges"]), (
        "Mismatch between positions and charges count."
    )
    assert len(start_params["positions"]) == len(end_params["positions"]), (
        "Start and end positions count must match."
    )


def get_param_label(start_params: dict, end_params: dict) -> str:
    """Generates text metadata for the plot, highlighting changed values."""
    label_lines = ["**Parameters**"]
    for key in start_params:
        v1, v2 = start_params[key], end_params[key]

        if key == "positions":
            label_lines.append("positions:")
            for i, (p1, p2) in enumerate(zip(v1, v2)):
                p1_arr, p2_arr = np.array(p1), np.array(p2)
                if np.array_equal(p1_arr, p2_arr):
                    label_lines.append(
                        f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1_arr)}]",
                    )
                else:
                    label_lines.append(
                        f"  P{i}: [{', '.join(f'{x:.1f}' for x in p1_arr)}] → [{', '.join(f'{x:.1f}' for x in p2_arr)}]",
                    )
        elif key == "charges":
            continue  # Skipped for brevity, but can be added similarly
        elif v1 == v2:
            label_lines.append(f"{key}: {v1}")
        else:
            label_lines.append(f"{key}: {v1} → {v2}")
    return "\n".join(label_lines)


def run_lerp_plot(
    system,
    start_params,
    end_params,
    lerp_step_count,
    plot_metrics,
    error_metrics,
):
    # 1. Validation & Initialization
    t_values = np.linspace(0, 1, lerp_step_count)
    results = {metric_key: [] for metric_key in plot_metrics}
    error_results = {metric_key: [] for metric_key in error_metrics}

    # 2. Execution Loop
    for t in t_values:
        system.electrostatics.clear()
        system.part.clear()
        params = lerp_dict(start_params, end_params, t)

        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])

        # Calculate Energies
        for key, eval_func in plot_metrics.items():
            results[key].append(eval_func(system, params))

        # Calculate Errors
        for key, eval_func in error_metrics.items():
            error_results[key].append(eval_func(system, params))

    # 3. Plot Generation
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create secondary y-axis

    # Plot Energies (Primary axis)
    for (label, mpl_tuple), values in results.items():
        ax1.plot(t_values, values, label=label, **dict(mpl_tuple))

    # Plot Errors (Secondary axis)
    for (label, mpl_tuple), values in error_results.items():
        ax2.plot(t_values, values, label=label, **dict(mpl_tuple))

    # Formatting
    ax1.set_xlabel(r"Interpolation Parameter $t$")
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Error |Custom - Reference|", color="purple")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.text(
        0.83,
        0.5,
        get_param_label(start_params, end_params),
        verticalalignment="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    save_plot_with_timestamp(fig)
    plt.show()
