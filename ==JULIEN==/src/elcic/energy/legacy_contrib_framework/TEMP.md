$E_{\text{total}} = \frac{1}{2}\left(\Phi(L_i, L_0) + \Phi(L_0, L_0)\right)$

$E_{\text{total}} = \frac{1}{2}\Phi(L_0, L_T) + \frac{1}{2}\Phi(L_0, L_{\pm2})$ = 1/2\*_E_\_near + 1/2\*E\_far

E_near_ _$= \Phi(L_0, L_T) = \frac{1}{2}\left(\Phi(L_T, L_T) - \Phi(L_{\pm1}, L_{\pm1}) + \Phi(L_0, L_0)\right)$_ _= 1/2 \* ( a - b + E_\_real\_self)

E\_far = $\Phi(L_0, L_{\pm2})$ (computed with far formula)





The Electrostatic Layer Correction with Image Charges (ELCIC) method computes the total electrostatic energy ($E_{\text{total}}$) of a $2D+h$ slab system bounded by two planar dielectric interfaces by combining several distinct contributions.

The total electrostatic energy of the real charges in the primary layer ($L_0$) interacting with each other and with the infinite series of polarization image charges ($L_i$) is initially defined as:

$E_{\text{total}} = \frac{1}{2}\left(\Phi(L_i, L_0) + \Phi(L_0, L_0)\right)$

where $\Phi(L, L')$ represents the electrostatic interaction between two charge groups under periodic boundary conditions in the $x$ and $y$ directions.

To evaluate this efficiently, the infinite image charges are split into near and far groups based on their distance to the primary simulation box:

* **Near image charges ($L_{\pm1}$):** This includes image charges in the top ($L_{+1}$) and bottom ($L_{-1}$) dielectrics that lie within a threshold distance $\lambda$ from the simulation box boundaries.
* **Far image charges ($L_{\pm2}$):** This includes the remaining image charges ($L_{+2}$ in the top and $L_{-2}$ in the bottom dielectrics) that are located further than the threshold distance $\lambda$.

***

### 1. Breakdown of the Energy Contributions

The total energy is divided into two primary parts based on this spatial grouping:

$E_{\text{total}} = \frac{1}{2}\Phi(L_0, L_T) + \frac{1}{2}\Phi(L_0, L_{\pm2})$

where $L_T = L_{-1} \cup L_0 \cup L_{+1}$ is the combined set of real charges and near image charges.

#### Contribution A: Far-Field Image Charge Interaction ($\Phi(L_0, L_{\pm2})$)

* This contribution handles the direct interaction between the real charges ($L_0$) and the distant image charges ($L_{\pm2}$).
* Because these charges are separated by a minimum distance of $\lambda$, this interaction is calculated directly using a fast "far formula" expansion.
* The infinite summation over the multiple polarization reflections forming $L_{\pm2}$ is performed analytically via geometric series, allowing this contribution to be computed with linear $\mathcal{O}(N)$ scaling.

#### Contribution B: Near-Field and Real Charge Interaction ($\Phi(L_0, L_T)$)

* This contribution handles the interactions of the real charges with themselves ($\Phi(L_0, L_0)$) and with the nearby image charges ($\Phi(L_0, L_{\pm1}$)).
* Standard 3D periodic Coulomb solvers (such as Particle-Particle Particle-Mesh, or P3M) cannot compute the cross-interaction $\Phi(L_0, L_T)$ directly; they can only evaluate the total internal self-interaction of a full set of charges, $\Phi(L_T, L_T)$.
* To resolve this, $\Phi(L_0, L_T)$ is mathematically decomposed into a combination of three distinct internal intra-group interactions:

$\Phi(L_0, L_T) = \frac{1}{2}\left(\Phi(L_T, L_T) - \Phi(L_{\pm1}, L_{\pm1}) + \Phi(L_0, L_0)\right)$

***

### 2. Computing the Intra-Group Terms via 3D Solver + ELC

Each of the three intra-group terms ($\Phi(L_T, L_T)$, $\Phi(L_{\pm1}, L_{\pm1})$, and $\Phi(L_0, L_0)$) describes a system that is periodic in only two dimensions ($2D+h$). They are computed by mapping them to a 3D fully periodic solver combined with an Electrostatic Layer Correction (ELC) term:

$\Phi_{2D+h}(L, L) = \Phi_{3D}(L, L) + E_{\text{ELC}}(L)$

Because the subsets $L_T$ and $L_{\pm1}$ are generally not charge-neutral (even if the original real charge set $L_0$ is neutral), the ELCIC method utilizes the **non-neutral extension of the ELC method**. This involves:

* Enclosing the system in an artificial 3D simulation box with a height expanded by an empty gap of $3\lambda$ ($L_z = l_z + 3\lambda$) to decouple unwanted periodic replicas along the $z$-axis.
* Introducing a homogeneous neutralizing background into the 3D periodic method (e.g., P3M) to handle the net charge of the subset safely.
* Utilizing a generalized ELC correction term that explicitly subtracts the unwanted interactions of the charges with the periodic replicas along the $z$-direction as well as the artificial neutralizing background.

***

### 3. Final Combination for $E_{\text{total}}$

By substituting the decomposed near-field identity into the primary energy equation, all contributions are combined into the final operational formula for the total electrostatic energy:

$E_{\text{total}} = \frac{1}{4}\left[\Phi(L_T, L_T) - \Phi(L_{\pm1}, L_{\pm1}) + \Phi(L_0, L_0)\right] + \frac{1}{2}\Phi(L_0, L_{\pm2})$

When expanded into the actual components evaluated by the algorithm, $E_{\text{total}}$ is obtained from:

1. **Three 3D Periodic Energy Calculations:** $\Phi_{3D}(L_T, L_T)$, $\Phi_{3D}(L_{\pm1}, L_{\pm1})$, and $\Phi_{3D}(L_0, L_0)$, typically computed using P3M.
2. **Three Non-Neutral ELC Corrections:** $E_{\text{ELC}}(L_T)$, $E_{\text{ELC}}(L_{\pm1})$, and $E_{\text{ELC}}(L_0)$ to remove the 3D replication and background artifacts.
3. **One Analytical Far-Field Summation:** $\Phi(L_0, L_{\pm2})$ computed via the product-decomposition far formula.

