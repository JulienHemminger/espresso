* setup LLM driven iterative diagnose & improvement pipeline

  * TODO Heres error over lz: err(lz=10) = 1e-3, err(lz=13) = 1e-4. The remaining parameters were fixed at ...

    * in the prompt

      * paper (tyagi...)
      * Find which contrib contains/causes the error (i.e. those depending on lz)
      * write a corrected code snippet

theres still E-near errors (i get err=1e-3, but e\_far can only account for 1e-5)

# Procedure

**Far-Field Isolation (Large Gap)**

Use `gap_size >> λ` (e.g., gap = 30, lz = 10) so near-field image charges are very distant.

Expected:`e_near` → 0 (images far away), energy dominated by `e_far`. If total energy is wrong here, the bug is in `_get_far_field_energy`.

**Near-Field Isolation (Tight Tolerance):** the systems ive been doing have efar=1e-5 and enear=4

# Action Tree

&#x20;&#x20;

* fix near energy
  * "a/b\_zshift\_" contains error spike at pos.z=lz/2

* fix far energy (e\_far = 1e-5) so my accuracy has to be like 1e-6 / 1e-8.

