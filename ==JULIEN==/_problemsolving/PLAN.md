TODO

param\_lerp\_plot\_2d: add running multiple lerps and plot them in one plot? do i really need this bulk-feature?

## aufarbeitung bisher elcic

* [start elcic, planning](https://github.com/JulienHemminger/espresso/commit/18dffc9de140c0b3b18ecdbd176b1962fac22abe)
* [impl ana method](https://github.com/JulienHemminger/espresso/commit/9159adef9638ed6bcd3bf4a87cd8ce66c98297b5) + entire branch with failed experiments
* [test: dipole shifting](https://github.com/JulienHemminger/espresso/commit/d596dbdae01ce9b449157e3a9291aa0bd2d20be8), [revisited later](https://github.com/JulienHemminger/espresso/commit/a532a816329deb1ef8fbb6b66a062e1c8290f230)
* [test: param sweep](https://github.com/JulienHemminger/espresso/commit/dcfc247873bc6f48140d1f1cc302f657adc3ef84)

  * found outliers in param sweep, create numgrad

## Make Progess

* TESTS are there so i can test/debug/anaylize individual contribs/components
*

- setup dual plate test, find an anasol if possible, extend elcic.py by "E-upper plate"

* setup non-neutral test, find an anasol if possible, extend elcic.py by E-non-neutral contrib

*

* custom\_elcic.py that

  * matches analytical (if available)
  * matches legacy\_elcic (except when particles near plate, there espresso has an error)

##

#

#### File structure (creating new tests and dependencies instead of modifying old ones)

elcic/single\_plate/neutral/metallic/

* analytical\_energy.py
* custom\_energy.py
* (legacy\_energy is shared)

elcic/single\_plate/neutral/non-metallic/

elcic/single\_plate/non-neutral/metallic/

elcic/single\_plate/non-neutral/non-metallic/       &#x20;

##### Notes

normally youd use git commits to "make sure no progess is thrown away"&#x20;
