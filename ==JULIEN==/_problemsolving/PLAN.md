How do i make progress?

* more structure
  * write test, modify test (until passed + all prev tests passed) then go on to next test
    * NEVER overwrite an existing test, etc.
    * try to do no code-sharing for tests (few depencencies that if modified can cause errors)

* make sure when i change things, it stays backw\.compatible
  * strict tests (assert...)

















Für rdm-param bekomme ich err=1e-5, 1e-6. Espressos ELC bekommt 1e-8, 1e-9
\* reicht das an genauigkeit? (es ist sehr aufwendig die quelle von Fehlern von nur 1e-6/7 zu finden und zu fixen)

ELCIC Implementation Checkpoints

* commit="err=1e-8 for large gap sizes, swapped "f\_max" formula" 1/1m err=1e-8 - BEST AT RDM PARAM (with part.Z >> 0)
* commit="fix elcic impl for no plates at all": 1/1, err=1e-1 err\_for\_near\_plate\_particles=0.4 - BEST AT Z-SHIFTING (part.Z -> 0)

AHEAD TESTING for rdm param

* "delta\_mid\_top": 0, "delta\_mid\_bot": -1, "charges": \[+1, -1], err=1e-5 (comparable to legacy)

* "delta\_mid\_top": -1, "delta\_mid\_bot": -1, "charges": \[+1, -1], err=1e-2 (comparable to legacy)
  * requires "const\_pot"=True in legacy\_elc\_energy.py

* "delta\_mid\_top": +1, "delta\_mid\_bot": -1, "charges": \[+1, -1], err=1e-1 (comparable to legacy)

* "delta\_mid\_top": +1, "delta\_mid\_bot": +1, "charges": \[+1, -1], err=1e-4 (better to legacy)

* "delta\_mid\_top": 0.3, "delta\_mid\_bot": -0.4, "charges": \[+1, -1], err=1e-6 (better to legacy)

* "delta\_mid\_top": 0.3, "delta\_mid\_bot": -0.4, "charges": \[+1, -1, -1], err=1e-5 (better to legacy)
  * no legacy energy: "ELC does not currently support non-neutral systems with a dielectric contrast.. Skipping..."

GRAINS OF SALT

* i compared it to the ana\_sol, idk how accurate the ana\_sol is
* did only a few samples per category

TODO how do i make sure i can actually "move on". i dont want to impl it in c++, realize sth isnt working and have to come back to the python prototype
