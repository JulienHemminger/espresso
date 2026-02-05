ACTION_DESCRIPTION, REWARD_ESTIMATE, CHILDREN, REFINEMENT = []
NO, MAYBE, YES_TODO, YES_DONE = []

f"""
* do "c++ implementation" directly: {NO}
    * {REWARD_ESTIMATE}
        * Con
            * more tideous to build & test, etc
            * more tideous to extend with ELCIC

* do "python prototype" -> "c++ implementation": {YES_TODO}
    * {REWARD_ESTIMATE}
        * Con
            * the "python -> c++" transition is difficult, wonky, etc. 
    * {REFINEMENT}
        * use elc.cpp as base: {NO}
            * {ACTION_DESCRIPTION}
                * refactor elc.cpp        
                    * write c++ test
                        * compare results: old_elc.cpp vs new_elc.cpp -also compare zwischenergebnisse
                        
                        * with (actor, gap_size, pw_error) signature

                    * simplyfy & refactor elc.cpp using code principles (encapsulation, KISS, DRY, ...)

                * write elc.py
                    * build framework/api similar to elc.cpp (same parameters, etc)
                    
                    * convert to python component-wise (einzelne code-blocks, zwischenergebnisse)

        * test-driven developemtn: {YES_TODO}
            * {ACTION_DESCRIPTION}
                * create "basic elc test" (minimal setup)
                * create elc algorithm that passes that test

                * create "medium elc test" (average setup)
                * create elc algoritm that passes that test and previous ones

                * ...
                * {REWARD_ESTIMATE}
                    * Pro
                        * i need to do these tests anyways
                        * doing "c++ -> python" first is cumbersome & maybe doesnt even work
"""

