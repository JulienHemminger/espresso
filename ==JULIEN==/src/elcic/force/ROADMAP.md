To gradually extend and rigorously test an Electrostatic Layer Correction (ELC) implementation into a full Electrostatic Layer Correction with Image Charges (ELCIC) method, it is best to construct a series of highly targeted, minimal physical systems. By isolating specific geometric zones, dielectric boundaries, and calculation paths, you can verify each component—particle classification, near-system ELC scaling, and analytical far-field sums—without bugs in one confounding the other.

Below is a structured roadmap of testing systems, moving from simple, uniform baselines to a fully functional ELCIC framework.

---

### Step 1: The Validation Baseline (Pure ELC)

Before touching any dielectric or image charge code, you must establish that your core 3D solver + ELC implementation behaves perfectly.

* **System Configuration:**
```python
params = {
    "lx": 50.0, "ly": 50.0, "gap_size": 20.0, "prefactor": 1.0,
    "delta_mid_top": 0.0,  # No reflection at top
    "delta_mid_bot": 0.0,  # No reflection at bottom
    "charges": [+1.0, -1.0],
    "pw_error": 1e-8,
    "positions": [np.array([25.0, 25.0, 15.0]), np.array([25.0, 25.0, 25.0])],
}
params["lz"] = params["gap_size"] + 40.0  # Total box height = 60.0

```


* **Target Goal:** The active region where particles reside is $z \in [0, \text{gap\_size}]$. By setting both reflection factors ($\Delta$) to `0.0`, the system must behave exactly like a regular ELC system.
* **Testing Focus:** Ensure your classification routine maps all particles into the central bulk group ($L_{0,0}$). The "Near" super-system $L_T$ should contain *only* the real particles ($L_0$), and the analytical far-field forces must evaluate identically to zero. Match these forces against an established 2D electrostatics reference (like MMM2D or a known working ELC code).

---

### Step 2: Isolating the Central Bulk ($L_{0,0}$) with Real Interfaces

Here you introduce dielectric jumps but place the particles deep inside the slab, keeping them away from the boundary layers where image-splitting occurs.

* **System Configuration:**
Assume your spatial classification threshold parameter is $\lambda = 5.0$.
```python
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
2.  **Near Force:** Because there are no boundary particles, no primary image layers $L_{+1}$ or $L_{-1}$ are generated. The "Near" ELC force calculation runs purely on the real particles $L_0$.
3.  **Far Force:** This is the ideal setup to test your analytical "far formula" implementation for the first time. The infinite series of images ($L_{\pm 2}, L_{\pm 3}, \dots$) will exert forces on these central particles. You can verify the analytical far forces by comparing them against an explicit, brute-force direct summation of image layers up to a high manual cutoff (e.g., tracking the first 100 image layers explicitly in space).

---

### Step 3: Single Interface Image Generation (Testing $L_{0,+1}$ and Top Forces)

Now you activate particle shifting and the creation of explicit primary image charges, but only at one boundary.

* **System Configuration:**
Set $\lambda = 5.0$. Place one particle close to the top interface ($z = \text{gap\_size}$).
```python
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



---

### Step 4: Symmetric Single Interface Image Generation (Testing $L_{0,-1}$ and Bottom Forces)

This mirrors Step 3 but targets the opposite boundary to catch any sign errors, coordinate orientation bugs, or indexing flaws in your spatial shifts.

* **System Configuration:**
Set $\lambda = 5.0$. Place a particle close to the bottom interface ($z = 0$).
```python
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



---

### Step 5: The Full Asymmetric System (Stress-Testing the Coupling)

The final stage combines all previous features simultaneously. Particles occupy all three spatial zones, and both boundaries have distinct dielectric mismatches.

* **System Configuration:**
Set $\lambda = 5.0$.
```python
params = {
    "lx": 50.0, "ly": 50.0, "gap_size": 20.0, "prefactor": 1.0,
    "delta_mid_top": 0.4,
    "delta_mid_bot": -0.8,
    "charges": [+1.0, -1.0, +0.5],
    "pw_error": 1e-8,
    "positions": [
        np.array([5.0, 5.0, 1.2]),    # Maps to L_0,-1 (Near bottom)
        np.array([25.0, 25.0, 11.0]), # Maps to L_0,0  (Bulk)
        np.array([45.0, 45.0, 19.1]), # Maps to L_0,+1 (Near top)
    ],
}
params["lz"] = params["gap_size"] + 40.0

```


* **Target Goal:** Evaluate the complete ELCIC force pipeline where near-field image generation, bulk interactions, and the complete infinite-series far formula happen at the same time.
* **Testing Focus:**
* Verify that the total super-system $L_T$ has exactly 5 particles (3 real + 1 top image + 1 bottom image).
* Verify that the analytical far formula seamlessly computes the cross-talk forces between the infinite image trains of the top particle acting on the bottom particle, and vice versa.
* **Self-Consistency Check (The Shift Test):** If you shift your entire system configuration upward in the $z$-direction by a small constant (while keeping the boundary planes locked relative to the particles), the physical forces must remain absolutely invariant. If the forces change when the absolute coordinates change, you have an indexing or origin-matching bug in your analytical far-field layer loops.