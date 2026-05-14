

## ask gemini which E-contribs "activate" for what systems. can i find a system where custom\_elcic needs fewer contribs?



* claudes



# Make Progess

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

#####
