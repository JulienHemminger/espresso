

# Procedure

* test wide range of symmetrical systems, assert `e_near_top ≈ e_near_bot`
* with delta mid bot = 0,  err e\_near\_bot=0 (also with top)

**Far-Field Isolation (Large Gap)**

Use `gap_size >> λ` (e.g., gap = 30, lz = 10) so near-field image charges are very distant.

Expected:`e_near` → 0 (images far away), energy dominated by `e_far`. If total energy is wrong here, the bug is in `_get_far_field_energy`.

**Near-Field Isolation (Tight Tolerance):** the systems ive been doing have efar=1e-5 and enear=4

# Action Tree

&#x20;&#x20;

* fix near energy
  * "a/b\_zshift\_" contains error spike at pos.z=lz/2

* fix far energy (e\_far = 1e-5) so my accuracy has to be like 1e-6 / 1e-8.

