import numpy as np
import matplotlib.pyplot as plt

def get_elc_energy(p3m, gap_size, pw_error, system, do_contributions_plot=True):
    
    lx, ly, lz = system.box_l
    parts = system.part.all()
    qs, (xs, ys, zs) = parts.q, parts.pos.T
    
    # 1. 3D Periodic Energy from P3M
    system.electrostatics.solver = p3m
    e_3d = system.analysis.energy()["total"]
    prefactor = p3m.prefactor
    
    # 2. Charge Moments
    xi0 = np.sum(qs)        
    xi1 = np.sum(qs * zs)   
    xi2 = np.sum(qs * zs**2)
    
    # 3. Handle the Non-Neutral/Dipole Corrections
    volume = lx * ly * lz
    fac = 2.0 * np.pi / volume
    
    # correction for non-neutral systems:
    e_non_neutral_corr = fac * (xi1**2 - xi0 * xi2 - (lz**2 / 12.0) * xi0**2)
    
    # 4. Reciprocal Space ELC Term
    f_max = -np.log(pw_error) / (2.0 * np.pi * gap_size)
    p_max = int(np.ceil(f_max * lx))
    q_max = int(np.ceil(f_max * ly))
    
    p = np.arange(-p_max, p_max + 1)
    q = np.arange(-q_max, q_max + 1)
    P, Q = np.meshgrid(p, q)
    P, Q = P.flatten(), Q.flatten()
    
    # Exclude the k=0 mode (handled by the real space and dipole terms)
    mask = ((P != 0) | (Q != 0)) & (np.sqrt((P/lx)**2 + (Q/ly)**2) <= f_max)
    fx, fy = P[mask] / lx, Q[mask] / ly
    f = np.sqrt(fx**2 + fy**2)

    
    # Particle-wise components
    arg_x, arg_y, arg_z = 2.0 * np.pi * fx, 2.0 * np.pi * fy, 2.0 * np.pi * f
    cx, sx = np.cos(arg_x * xs[:, None]), np.sin(arg_x * xs[:, None])
    cy, sy = np.cos(arg_y * ys[:, None]), np.sin(arg_y * ys[:, None])
    ex_p, ex_m = np.exp(arg_z * zs[:, None]), np.exp(-arg_z * zs[:, None])

    # Compute form factors (Chi) linearly
    def s_term(ez, c1, c2): 
        return np.sum(qs[:, None] * ez * c1 * c2, axis=0)

    # Summing over the four combinations of sin/cos for the 2D Fourier transform
    chi = (s_term(ex_p, cx, cy) * s_term(ex_m, cx, cy) + 
           s_term(ex_p, sx, cy) * s_term(ex_m, sx, cy) +
           s_term(ex_p, cx, sy) * s_term(ex_m, cx, sy) + 
           s_term(ex_p, sx, sy) * s_term(ex_m, sx, sy))
    
    # The reciprocal energy correction
    rep = np.exp(-arg_z * lz) / (1.0 - np.exp(-arg_z * lz))
    e_recip = -np.sum((1.0 / (lx * ly * f)) * rep * chi)

    # Plotting
    if do_contributions_plot:
        show_contributions_plot(prefactor, e_recip, e_3d, e_non_neutral_corr)
    
    # 5. Final Energy Assembly
    return e_3d + (prefactor * e_non_neutral_corr) + (prefactor * e_recip)

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