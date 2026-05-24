# Procedure

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







* impl 3d lerp plot

  * use it to lerp over a grid of delat\_mit bot x top

