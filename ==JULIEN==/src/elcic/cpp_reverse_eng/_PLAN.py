YES = None
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

* recreate elc.cpp from code + output: TODO
    * REFINEMENT
        * clean/simplify elc.cpp: {NO}
            * Con:
                * i cant simplify that much (around 150lines out of 1.3k. maybe a little sin/cos cache, etc)
                * testing (build w cmake) always takes long 


        * just parse contrib-wise to python code: {YES}
            * re-create  E_return_of(elc.cpp > ElectrostaticLayerCorrection::long_range_energy()) = ('coulomb', 1)
                
                * print long_range_energy()'s input parameters TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo
                * create py template that gets same input parameters
                * von zw.ergebnis zu zw.ergbenis baue long_range_energy() in py nach
                    
                    * ??? aber es wird wahrsch. tiefe dependencies/vernestung (e.g. ich müsste p3m.cpp auch nachbauen oder so) geben ??????????????????????ßßß
                    * ??? vllt pro zw.ergebnis entscheiden ob ich "code nachbauen" oder "output matchen" mache

            * re-create ('coulomb', 0) and find out where its from, is only 2e-8

            


    * CHILDREN       
        * fix the error inherited from elc.cpp(when particles near the diel.interfaces?)
        * extend it for non-neutral, non/metallic where legacy elc doesnt work)

        * do the same with forces

"""


