from collections.abc import Callable
from typing import Any

import espressomd
import espressomd.electrostatics
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
    system: espressomd.System,
    start_params: dict,
    end_params: dict,
    lerp_step_count: int,
    plot_metrics: dict[
        tuple[str, tuple[tuple[str, Any], ...]],
        Callable[[espressomd.System, dict], float],
    ],
):
    """
    Evaluates dynamic system properties across a linear parameter space and plots them.

    :param plot_metrics: Dict mapping ((label_string, ((param_name, value), ...))) -> function(system, params)
    """
    # 1. Validation
    validate_inputs(start_params, end_params, lerp_step_count)

    # 2. Initialization
    t_values = np.linspace(0, 1, lerp_step_count)
    results = {metric_key: [] for metric_key in plot_metrics}

    # 3. Execution Loop
    for t in t_values:
        system.electrostatics.clear()
        system.part.clear()

        params = lerp_dict(start_params, end_params, t)
        system.box_l = [params["lx"], params["ly"], params["lz"]]
        for i in range(len(params["charges"])):
            system.part.add(pos=params["positions"][i], q=params["charges"][i])

        for (label, mpl_tuple), eval_func in plot_metrics.items():
            value = eval_func(system, params)
            results[(label, mpl_tuple)].append(value)

    # 4. Plot Generation
    fig, ax = plt.subplots(figsize=(10, 6))

    for (label, mpl_tuple), values in results.items():
        # Convert the immutable tuple back into a dictionary for matplotlib
        mpl_params = dict(mpl_tuple)
        ax.plot(t_values, values, label=label, **mpl_params)

    ax.set_xlabel(r"Interpolation Parameter $t$")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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
