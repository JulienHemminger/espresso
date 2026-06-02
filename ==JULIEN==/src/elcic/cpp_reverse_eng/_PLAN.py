
DONE, NO = None, None
f"""
Action Tree
* fix custom.py for this single test: {DONE}
* add range of test (lerp) and plots (now im at my old err=1e-4): {DONE}
* check if the way elc.cpp splits up contribs(Etotal=Enear+Efar. Enear=Enearcorr+Eneardipole) is correct? {DONE}
* check if the elc-cpp contribs are correct (or is e.g. error in Enear balancing the error in Efar) {DONE}


* i could to "by hand": setup lerp lz test, find errors (in indiv contribs just like i did for total energy): {NO}, still very costly



Action Tree
* recreate elc.cpp from output only: {NO}
    * REWARD_ESTIM
        * Con
            * mapping elc.cpp contribs to py contribs is difficult, wonky, shaky

* map individual contribs/methods from elc.cpp to custom.py
    * fix E_far (elc.cpp: "<< ", E_far = E_far_p3m + E_far_corr = " << total") TODO

* recreate elc.cpp from code + output: TODO
    * REFINEMENT
        * clean/simplify elc.cpp?
        * just write py code?

    * CHILDREN
        * 



Action Tree
* recreate in python elc.cpp from code + output
* fix the error inherited from elc.cpp(when particles near the diel.interfaces?)
* extend it for non-neutral, non/metallic where legacy elc doesnt work)

* do the same with forces



??????????????????????????
* find the entry point when i call espresso.ELC in python

* print contribs and params on C++ side

* one by one, recreate contribs in python unil i have working custom.py
"""


