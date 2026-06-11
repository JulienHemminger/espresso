$\mathbf{F}_i = \mathbf{F}_i^{\text{3D base}} + \mathbf{F}_i^{\text{ELC (no dielectrics)}} + \mathbf{F}_i^{\text{near images}} + \mathbf{F}_i^{\text{far images}}$

* regular ELC

  * params: delta\_mid\_bot=+0.0, delta\_mid\_top=+0.0
* near

  * how do i do this?
  * single plate (single reflection?)
  * dual plates (repeated reflections)
* far - interactions between L0 and L+-2

what do i do differntly than this morning?

* closer to the paper (no 5 msgs betwen paper and code in LLM chat)

do idxbulk, idxtop and idxbot get treated differently?

\================

***

### Step 1: Pure ELC, no plates): DONE

### Step 2: z shifting, soft-check masking: DONE

### Step 3: lower plate only, hard-coded reflection

​charges: qi​\*Δb​, qi\*Δb\*​Δ, qi\*​Δb​\*Δ²,

positions: −zi​, −(2lz​−zi​), −(4lz​−zi​),…

### Step 4: Full $F_{\text{near}}$

Define the “near-system” including real charges and near images:

$L_T = L_{-1} \cup L_0 \cup L_{+1}$

This is again a 2D+h system (possibly non-neutral). To evaluate interactions of real charges with all charges in $L_T$:

1. **Embed** **$L_T$** **in an enlarged 3D periodic box:**

$L_x = l_x,\quad L_y = l_y,\quad L_z = l_z + 3\lambda$

so a free gap of thickness $\lambda$ remains at the top and bottom.
2\. **Compute energy and forces** for $L_T$ in this 3D box with a standard periodic Coulomb method (e.g., P3M) plus the non-neutral ELC correction \[Eq. (3.10)]:

$E_{2D+h}(L_T) = E_{3D}(L_T) + E_{\text{ELC}}(L_T)$

and obtain forces $\mathbf{F}_i^{(L_T)}$ on all charges $i \in L_T$.
3\. **To isolate the contribution acting on real charges only**, note that:

$\langle \text{energy of }L_0\text{ with }L_T\rangle = \frac{1}{2} \left[ E(L_T,L_T) - E(L_{+1}\cup L_{-1},L_{+1}\cup L_{-1}) + E(L_0,L_0) \right]$

with each term again obtained as a 3D+ELC evaluation in appropriate boxes.
4\. **For forces**, simply discard forces on image charges; the physical forces on real charges are:

$\mathbf{F}_i^{\text{near}} = \mathbf{F}_i^{(L_T)},\qquad i\in L_0$

L0​–L±1L±1​ + L0L0​–L0L0​ to be handled via a standard 3D solver + ELC (near interactions);

L0​–L±2L±2​ to be handled by the far‑formula factorization.



Step 5: F\_far

***

### Step 2: Isolating the Central Bulk ($L_{0,0}$) with Real Interfaces

Here you introduce dielectric jumps but place the particles deep inside the slab, keeping them away from the boundary layers where image-splitting occurs.

* **System Configuration:**
  Assume your spatial classification threshold parameter is $\lambda = 5.0$.

```Python
params = {
    "lx": 50.0, "ly": 50.0, "gap_size": 20.0, "prefactor": 1.0,
    "delta_mid_top": 0.8,   # Active top boundary
    "delta_mid_bot": -0.5,  # Active bottom boundary
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([25.0, 25.0, 10.0]),  # Z is exactly in the middle of gap_size
        np.array([10.0, 10.0, 10.0])   # Safely within [lambda, gap_size - lambda]
    ],
}
params["lz"] = params["gap_size"] + 40.0

```

* **Target Goal:** Test the execution path where no real particles are near the boundaries, meaning $L_{0,+1}$ and $L_{0,-1}$ are empty.
* **Testing Focus:** 1.  **Classification:** Verify that both particles are classified as $L_{0,0}$.

1. **Near Force:** Because there are no boundary particles, no primary image layers $L_{+1}$ or $L_{-1}$ are generated. The "Near" ELC force calculation runs purely on the real particles $L_0$.
2. **Far Force:** This is the ideal setup to test your analytical "far formula" implementation for the first time. The infinite series of images ($L_{\pm 2}, L_{\pm 3}, \dots$) will exert forces on these central particles. You can verify the analytical far forces by comparing them against an explicit, brute-force direct summation of image layers up to a high manual cutoff (e.g., tracking the first 100 image layers explicitly in space).

***

### Step 3: Single Interface Image Generation (Testing $L_{0,+1}$ and Top Forces)

Now you activate particle shifting and the creation of explicit primary image charges, but only at one boundary.

* **System Configuration:**
  Set $\lambda = 5.0$. Place one particle close to the top interface ($z = \text{gap\_size}$).

```Python
params = {
    "lx": 50.0, "ly": 50.0, "gap_size": 20.0, "prefactor": 1.0,
    "delta_mid_top": 0.6,   # Positive reflection
    "delta_mid_bot": 0.0,   # Keep bottom turned off to isolate errors
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([25.0, 25.0, 18.5]), # At z=18.5, distance to top (20) is 1.5 < lambda
        np.array([15.0, 15.0, 10.0])  # Remains in the bulk (L_0,0)
    ],
}
params["lz"] = params["gap_size"] + 40.0

```

* **Target Goal:** Force the algorithm to create a shifted primary image charge above the top layer.
* **Testing Focus:**

1. **Classification:** The particle at $z=18.5$ must be classified into $L_{0,+1}$.
2. **Super-system Expansion:** Check that your code populates the temporary system $L_T$ with the two real particles *plus* one virtual image particle positioned at $z_{\text{image}} = 2 \cdot \text{gap\_size} - z_i = 21.5$. The charge of this virtual particle must be correctly scaled to $q_{\text{image}} = \Delta_{\text{mid\_top}} \cdot q_i$.
3. **Force Filtering:** Ensure that after the ELC solver runs on $L_T$, the forces calculated on the virtual image particle are completely discarded and do not pollute your real particle arrays.
4. **Far Formula Modification:** Verify that the analytical far-field code properly accounts for $\Delta_{\text{mid\_bot}} = 0$, causing specific branches of the geometric structure factors ($\chi, \xi$) to collapse or simplify cleanly.

***

### Step 4: Symmetric Single Interface Image Generation (Testing $L_{0,-1}$ and Bottom Forces)

This mirrors Step 3 but targets the opposite boundary to catch any sign errors, coordinate orientation bugs, or indexing flaws in your spatial shifts.

* **System Configuration:**
  Set $\lambda = 5.0$. Place a particle close to the bottom interface ($z = 0$).

```Python
params = {
    "lx": 50.0, "ly": 50.0, "gap_size": 20.0, "prefactor": 1.0,
    "delta_mid_top": 0.0,   # Keep top turned off
    "delta_mid_bot": -0.7,  # Negative reflection at bottom
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [
        np.array([25.0, 25.0, 1.5]),  # Distance to bottom (0) is 1.5 < lambda
        np.array([15.0, 15.0, 10.0])  # Remains in the bulk
    ],
}
params["lz"] = params["gap_size"] + 40.0

```

* **Target Goal:** Force the algorithm to create a shifted primary image charge below the bottom layer.
* **Testing Focus:**

1. **Classification:** The particle at $z=1.5$ must map to $L_{0,-1}$.
2. **Super-system Expansion:** Check that the virtual particle added to $L_T$ is located at $z_{\text{image}} = -z_i = -1.5$, with a charge scaled by $\Delta_{\text{mid\_bot}}$.
3. **Anisotropic Validation:** Ensure your box padding adjustments in the 3D solver can handle negative coordinates safely if your ELC implementation requires shifting all coordinates to a positive domain before feeding them to the 3D grid solver (e.g., P3M).

***

### Step 5: full param sweep

* needs to be invariant to tuning param $\lambda$

