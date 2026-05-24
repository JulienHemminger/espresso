merge my "param\_lerp\_plot\_2d.py"

plot contribs in 2d lerp plot

impl 2d lerp batch plot

impl 3d lerp plot, use it to lerp over a grid of delat\_mit bot x top

impl old tests (no plate, single plate)

# Procedure - isnt there a better way?

* z-shift test -> test masking, depending on part.z ALL particles are either in L0, L+1 or L-1
* test wide range of symmetrical systems, assert `e_near_top ≈ e_near_bot`
* with `delta``midbot = 0`_, errnear_bot=0 (also with top)

**Far-Field Isolation (Large Gap)**

Use `gap_size >> λ` (e.g., gap = 30, lz = 10) so near-field image charges are very distant.

Expected:`e_near` → 0 (images far away), energy dominated by `e_far`. If total energy is wrong here, the bug is in `_get_far_field_energy`.

**Near-Field Isolation (Tight Tolerance):** the systems ive been doing have efar=1e-5 and enear=4





specify lerp params (e.g. lerp over delta\_mid\_bot)

* improve custom.py until err=1e-6

specify lerp params (e.g. lerp over delta*midtop*)

* improve custom.py until err=1e-6

# Action Tree

* fix custom.py structure

  * old\_plot.py, old\_custom.py
  * new\_custom.py

&#x20;&#x20;

* do i need to test old\_plot.py against old\_custom.py AND new\_custom.py

* GOAL: one custom.py that passes all the tests.

* fix near energy
  * "b\_zshift\_both\_non\_metallic" contains error spike at pos.z=lz/2

* fix far energy (e\_far = 1e-5) so my accuracy has to be like 1e-6 / 1e-8.

