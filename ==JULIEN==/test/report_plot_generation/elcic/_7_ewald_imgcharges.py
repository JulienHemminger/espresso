import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfc


def generate_image_charges(pos, q, lz, delta_b, delta_t, n_img):
    """Generates image charge positions and magnitudes up to reflection order n_img

    for dielectric boundaries at z = 0 (delta_b) and z = lz (delta_t).
    """
    img_pos = []
    img_q = []
    delta = delta_b * delta_t

    for m in range(n_img + 1):
        factor = delta**m

        # Series 1 (Bottom boundary): q * delta^m * delta_b at z = -2*m*lz - z_i
        if abs(delta_b) > 1e-12 and abs(factor * delta_b) > 1e-15:
            p = pos.copy()
            p[:, 2] = -2 * m * lz - pos[:, 2]
            img_pos.append(p)
            img_q.append(q * factor * delta_b)

        # Series 1 (Top boundary): q * delta^m * delta_t at z = 2*(m+1)*lz - z_i
        if abs(delta_t) > 1e-12 and abs(factor * delta_t) > 1e-15:
            p = pos.copy()
            p[:, 2] = 2 * (m + 1) * lz - pos[:, 2]
            img_pos.append(p)
            img_q.append(q * factor * delta_t)

        if m > 0:
            # Series 2 (Bottom boundary): q * delta^m at z = -2*m*lz + z_i
            if abs(factor) > 1e-15:
                p_bot = pos.copy()
                p_bot[:, 2] = -2 * m * lz + pos[:, 2]
                img_pos.append(p_bot)
                img_q.append(q * factor)

                # Series 2 (Top boundary): q * delta^m at z = 2*m*lz + z_i
                p_top = pos.copy()
                p_top[:, 2] = 2 * m * lz + pos[:, 2]
                img_pos.append(p_top)
                img_q.append(q * factor)

    if len(img_pos) > 0:
        return np.vstack(img_pos), np.hstack(img_q)
    else:
        return np.empty((0, 3)), np.empty((0,))


from scipy.special import erfcx


def get_ewald_energy_2d_dielectric(
    pos_real,
    q_real,
    box_l,
    delta_b=0.0,
    delta_t=0.0,
    n_max=30,
    n_img=5,
    prefactor=1.0,
):
    pos_real = np.asarray(pos_real, dtype=np.float64)
    q_real = np.asarray(q_real, dtype=np.float64)
    lx, ly, lz = box_l[0], box_l[1], box_l[2]
    area = lx * ly
    eta = np.sqrt(np.pi) / min(lx, ly)

    # 1. Generate image charges
    pos_img, q_img = generate_image_charges(
        pos_real, q_real, lz, delta_b, delta_t, n_img
    )

    if len(q_img) > 0:
        pos_all = np.vstack([pos_real, pos_img])
        q_all = np.hstack([q_real, q_img])
    else:
        pos_all = pos_real
        q_all = q_real

    n_real = len(q_real)
    n_total = len(q_all)

    # FIX: Displacement vectors between ALL charges (real + image)
    dr = pos_all[:, None, :] - pos_all[None, :, :]
    q_pairs = q_all[:, None] * q_all[None, :]

    # 2. Real Space Contribution
    grid_range = np.arange(-n_max, n_max + 1)
    nx, ny = np.meshgrid(grid_range, grid_range, indexing="ij")
    rx_shifts = nx.flatten() * lx
    ry_shifts = ny.flatten() * ly

    e_real = 0.0
    for rx, ry in zip(rx_shifts, ry_shifts):
        r_vec = dr + np.array([rx, ry, 0.0])
        dist = np.linalg.norm(r_vec, axis=2)

        if rx == 0.0 and ry == 0.0:
            # Mask self-interaction for real charges
            np.fill_diagonal(dist, np.inf)

        e_real += np.sum(q_pairs * erfc(eta * dist) / dist)
    e_real *= 0.5

    # 3. Reciprocal Space Contribution (using all charges)
    kx_base = 2.0 * np.pi / lx
    ky_base = 2.0 * np.pi / ly
    mx, my = np.meshgrid(grid_range, grid_range, indexing="ij")
    mx, my = mx.flatten(), mx.flatten()

    valid_g = (mx != 0) | (my != 0)
    gx = mx[valid_g] * kx_base
    gy = my[valid_g] * ky_base
    g = np.sqrt(gx**2 + gy**2)

    dr_xy = dr[:, :, :2]
    dz = dr[:, :, 2]

    e_recip = 0.0
    for k in range(len(g)):
        phase = dr_xy[:, :, 0] * gx[k] + dr_xy[:, :, 1] * gy[k]
        arg_plus = g[k] / (2.0 * eta) + eta * dz
        arg_minus = g[k] / (2.0 * eta) - eta * dz

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            term1 = np.exp(np.clip(g[k] * dz - arg_plus**2, -500, 500)) * erfcx(
                np.clip(arg_plus, -25, 25)
            )
            term2 = np.exp(np.clip(-g[k] * dz - arg_minus**2, -500, 500)) * erfcx(
                np.clip(arg_minus, -25, 25)
            )
            h_g = (
                np.exp(-1.0 * np.clip(arg_plus**2, 0, 700)) * term1
                + np.exp(-1.0 * np.clip(arg_minus**2, 0, 700)) * term2
            )
            h_g = np.nan_to_num(h_g, nan=0.0, posinf=0.0, neginf=0.0)

        e_recip += np.sum(q_pairs * (np.pi / g[k]) * h_g * np.cos(phase))
    e_recip /= 2.0 * area

    # 4. Self Energy (Computed over all charges included in the sum)
    e_self = -(eta / np.sqrt(np.pi)) * np.sum(q_all**2)

    # 5. G = 0 Term
    abs_dz = np.abs(dz)
    g0_terms = np.where(
        abs_dz < 1e-15,
        1.0 / (eta * np.sqrt(np.pi)),
        abs_dz * erf(eta * abs_dz)
        + np.exp(-((eta * abs_dz) ** 2)) / (eta * np.sqrt(np.pi)),
    )
    e_k0 = -(np.pi / area) * np.sum(q_pairs * g0_terms)

    # Note: A factor of 1/2 or specific image-counting corrections
    # are standard depending on convention, but this sum now properly includes
    # image-image interactions.
    return prefactor * (e_real + e_recip + e_k0 + e_self)


# =====================================================================
# Demonstration & Convergence Plot
# =====================================================================
from src.common.plot_saving import save_plot_with_timestamp

if __name__ == "__main__":
    # Define test parameters
    params = {
        "lx": 10.0,
        "ly": 10.0,
        "lz": 20.0,
        "delta_mid_bot": 0.5,  # Bottom metallic interface
        "delta_mid_top": 0.5,  # Top dielectric interface
        "prefactor": 1.0,
        "charges": [+1.0, -1.0],
        "positions": [np.array([6.0, 5.0, 4.0]), np.array([3.0, 2.0, 1.0])],
    }

    pos = np.array(params["positions"])
    q = np.array(params["charges"])
    box_l = [params["lx"], params["ly"], params["lz"]]

    n_img_range = range(20 + 1)
    energies = []

    for n_img in n_img_range:
        E = get_ewald_energy_2d_dielectric(
            pos_real=pos,
            q_real=q,
            box_l=box_l,
            delta_b=params["delta_mid_bot"],
            delta_t=params["delta_mid_top"],
            n_max=25,
            n_img=n_img,  # Now a standard Python int
            prefactor=params["prefactor"],
        )
        energies.append(E)

    energies = np.array(energies)
    for n_img, E in zip(n_img_range, energies):
        print(f"N_img: {n_img} | Energy: {E:.12f}")
    e_exact = energies[-1]  # Reference asymptotic energy at high n_img
    energy_errors = np.abs(energies[:-1] - e_exact)

    # =====================================================================
    # Combined Plot with Secondary Y-Axis
    # =====================================================================
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left Y-Axis: Total Absolute Energy
    color_energy = "#1f77b4"
    ax1.set_xlabel("Number of Image Generations ($\\mathrm{N_{img}}$)", fontsize=11)
    ax1.set_ylabel("Energy", color=color_energy, fontsize=11)
    line1 = ax1.plot(
        n_img_range,
        energies,
        "o-",
        color=color_energy,
        linewidth=2,
        markersize=6,
        label="Absolute Energy",
    )
    ax1.axhline(
        e_exact,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Asymptotic Limit ($E_{\\mathrm{exact}}$)",
    )
    ax1.tick_params(axis="y", labelcolor=color_energy)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right Y-Axis: Convergence Error (Log scale)
    ax2 = ax1.twinx()
    color_error = "#d62728"
    ax2.set_ylabel(
        "Error",
        color=color_error,
        fontsize=11,
    )
    line2 = ax2.semilogy(
        n_img_range[:-1],
        energy_errors,
        "s--",
        color=color_error,
        linewidth=2,
        markersize=6,
        label="$\\mathrm{|E(N_{img}) - E_{converged}|}$",
    )
    ax2.tick_params(axis="y", labelcolor=color_error)
    ax2.grid(False)  # Avoid conflicting grid lines on the secondary axis

    # Combine legends from both axes
    lines = line1 + [ax1.get_legend_handles_labels()[1][1]] + line2
    labels = [l.get_label() if hasattr(l, "get_label") else l for l in lines]
    # Alternatively, a clean manual legend:
    ax1.legend(
        [line1[0], line2[0]],
        [
            "$\\mathrm{E(N_{img})}$",
            "$\\mathrm{|E(N_{img}) - E_{converged}|}$",
        ],
        loc="center right",
    )

    plt.tight_layout()

    save_plot_with_timestamp(fig)
    plt.show()
