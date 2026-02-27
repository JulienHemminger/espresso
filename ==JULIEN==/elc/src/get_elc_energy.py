import numpy as np
import espressomd
import espressomd.electrostatics
import matplotlib.pyplot as plt

def get_elc_energy(p3m, gap_size, pw_error, system, do_contributions_plot=False):
    """
    Computes the ELC-corrected electrostatic energy for a slab geometry.
    Ensures stability for varying particle distributions and dipole moments.
    """
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, pos = parts.q, parts.pos
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # 1. Standard 3D P3M energy calculation (with PBC)
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    prefactor = p3m.prefactor
    vol = lx * ly * lz

    # 2. Corrected Dipole Term (Yeh-Berkowitz slab-wise summation correction)
    # The dipole correction is sensitive to the total dipole moment Mz.
    # The term (2 * pi / Volume) * Mz^2 is crucial for slab systems.
    mz = np.sum(qs * zs)
    e_corr_dipole = (2.0 * np.pi / vol) * (mz**2) * prefactor

    # 3. Dynamic Reciprocal Space Correction
    # Calculate required cutoff based on requested error (pw_error)
    # The convergence is exponential: exp(-2 * pi * f * gap_size)
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    
    # Generate frequencies, keeping P=0, Q=0 out to avoid singularities
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q, indexing='ij')
    P, Q = P.flatten(), Q.flatten()
    
    mask = (P != 0) | (Q != 0)
    P, Q = P[mask], Q[mask]
    
    fx, fy = P / lx, Q / ly
    f = np.sqrt(fx**2 + fy**2)
    
    # Filter by f_max
    valid = f <= f_max
    P, Q, f, fx, fy = P[valid], Q[valid], f[valid], fx[valid], fy[valid]

    # Compute Chi products using efficient outer products
    # This avoids nested Python loops and leverages BLAS
    arg_x = 2.0 * np.pi * fx[None, :] * xs[:, None]
    arg_y = 2.0 * np.pi * fy[None, :] * ys[:, None]
    
    cos_x, sin_x = np.cos(arg_x), np.sin(arg_x)
    cos_y, sin_y = np.cos(arg_y), np.sin(arg_y)
    
    # Pre-calculate exp terms for the two components
    arg_z = 2.0 * np.pi * f[None, :]
    exp_p = np.exp(zs[:, None] * arg_z)
    exp_m = np.exp(-zs[:, None] * arg_z)
    
    def get_sum(ez, cx, cy):
        return np.sum(qs[:, None] * ez * cx * cy, axis=0)

    # Compute S terms
    s_cc_p, s_cc_m = get_sum(exp_p, cos_x, cos_y), get_sum(exp_m, cos_x, cos_y)
    s_sc_p, s_sc_m = get_sum(exp_p, sin_x, cos_y), get_sum(exp_m, sin_x, cos_y)
    s_cs_p, s_cs_m = get_sum(exp_p, cos_x, sin_y), get_sum(exp_m, cos_x, sin_y)
    s_ss_p, s_ss_m = get_sum(exp_p, sin_x, sin_y), get_sum(exp_m, sin_x, sin_y)

    chi_prod = (s_cc_p * s_cc_m + s_sc_p * s_sc_m + 
                s_cs_p * s_cs_m + s_ss_p * s_ss_m)
    
    # Apply slab replica factor
    # This accounts for the infinite summation over slab images
    replica_factor = np.exp(-2.0 * np.pi * f * lz) / (1.0 - np.exp(-2.0 * np.pi * f * lz))
    
    e_corr_recip = -np.sum(((1.0 / (lx * ly)) / f) * replica_factor * chi_prod)
    

    # Plotting
    if do_contributions_plot:
        show_contributions_plot(prefactor, e_corr_recip, e_3d, e_corr_dipole)

    return e_3d + e_corr_dipole + (prefactor * e_corr_recip)



def show_contributions_plot(prefactor, e_corr_recip, e_3d, e_corr_dipole):
    # Define the components
        e_recip_final = prefactor * e_corr_recip
        components = {
            'P3M (3D)': e_3d,
            'Yeh-Berkowitz': e_corr_dipole,
            'ELC Reciprocal': e_recip_final
        }
        
        labels = list(components.keys())
        values = list(components.values())
        total_energy = e_3d + e_corr_dipole + e_recip_final

        fig, ax = plt.subplots(figsize=(9, 6))

        # 1. Stacked Bar for Contributions (Index 0)
        current_bottom = 0
        for i, (label, val) in enumerate(components.items()):
            ax.bar(0, val, bottom=current_bottom, label=label, edgecolor='white', width=0.6)
            
            # Use numeric x=0 for text placement
            if abs(val) > abs(total_energy * 0.05):
                ax.text(0, current_bottom + val/2, f'{val:.2f}', 
                        ha='center', va='center', fontweight='bold', color='white')
            current_bottom += val

        # 2. Total Energy Bar (Index 1)
        ax.bar(1, total_energy, color='gray', alpha=0.3, label='Final ELC Sum', width=0.6)
        ax.text(1, total_energy, f'{total_energy:.2f}', 
                ha='center', va='bottom' if total_energy > 0 else 'top', 
                fontweight='bold', color='black')

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Breakdown', 'Total Energy'])
        ax.set_ylabel('Energy (Reduced Units)')
        ax.set_title('ELC Electrostatic Energy Decomposition')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        # Move legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.show()