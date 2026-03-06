import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt
import numpy as np
import pytest
from elc.src.common.get_positions import get_rdm_constrained_points
from elc.src.common.has_downward_trend import has_downward_trend
from elc.src.energy.get_elc_energy import get_elc_energy_contribs
from elc.src.energy.third_party.get_ewald_energy_2d import get_ewald_energy_2d

# --- Helpers ---


def create_plot(pw_errors, elc_errors, contrib_data, title):
    """Encapsulates the complex plotting logic shared by both test cases."""
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    colors = ["#1abc9c", "#f1c40f", "#9b59b6"]
    bottoms = np.zeros(len(pw_errors))
    bar_width = 0.2 * np.array(pw_errors)

    # 1. Secondary Axis: Energy Contributions (Stacked Bars)
    for i, (label, vals) in enumerate(contrib_data.items()):
        ax2.bar(
            pw_errors,
            vals,
            bottom=bottoms,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.3,
            edgecolor="grey",
        )
        bottoms += np.array(vals)

    # 2. Primary Axis: Errors (Lines)
    ax1.loglog(
        pw_errors,
        elc_errors,
        "o-",
        label="ELC Error",
        color="#2980b9",
        linewidth=2,
        zorder=5,
    )

    ax1.loglog(pw_errors, pw_errors, "k:", alpha=0.5, label="Target Accuracy (1:1)")

    # Formatting
    ax1.set_xlabel("Requested Accuracy (pw_error)")
    ax1.set_ylabel("Measured Error (Log Scale)", color="#2980b9")
    ax2.set_ylabel("Energy Component Value (Linear Scale)", color="#7f8c8d")
    plt.title(title)

    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + bars, labels + bar_labels, loc="upper left", bbox_to_anchor=(1.15, 1)
    )

    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.invert_xaxis()
    fig.tight_layout()
    plt.show()


# --- Fixtures ---


@pytest.fixture(scope="module")
def es_system():
    system = espressomd.System(box_l=[10.0, 10.0, 3.0])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    yield system
    system.part.clear()


# --- Main Test ---


@pytest.mark.parametrize("system_type", ["neutral", "non-neutral"])
@pytest.mark.parametrize("show_convergence_plot", [True])
def test_accuracy_convergence(es_system, system_type, show_convergence_plot):
    # Setup parameters based on system type
    pw_errors = np.logspace(-4, -8, num=5)
    charges = [+1.0, -1.0]
    title = "Accuracy Convergence: " + system_type

    gap_size = 1.0
    lx, ly, lz = es_system.box_l
    pos1, pos2 = get_rdm_constrained_points(lx, ly, lz - gap_size - 1e-3)

    elc_errors = []
    contrib_data = {"P3M (3D)": [], "Yeh-Berkowitz": [], "ELC Reciprocal": []}

    for pw_err in pw_errors:
        es_system.part.clear()
        es_system.part.add(pos=pos1, q=charges[0])
        es_system.part.add(pos=pos2, q=charges[1])

        ana_energy = get_ewald_energy_2d(es_system, n_max=100)

        pref, e_recip, e_3d, e_non_neutral_corr = get_elc_energy_contribs(
            gap_size, pw_err, es_system
        )

        e_recip_final = pref * e_recip
        e_dipole_final = pref * e_non_neutral_corr
        elc_en = e_3d + e_dipole_final + e_recip_final

        # Data collection
        elc_errors.append(abs(elc_en - ana_energy))

        contrib_data["P3M (3D)"].append(e_3d)
        contrib_data["Yeh-Berkowitz"].append(e_dipole_final)
        contrib_data["ELC Reciprocal"].append(e_recip_final)

    # --- Assertions ---
    assert has_downward_trend(elc_errors)

    if show_convergence_plot:
        create_plot(pw_errors, elc_errors, contrib_data, title)
