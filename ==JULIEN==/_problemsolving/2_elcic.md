# Phase 1: Energy Implementation & Testing

## Single Plate

#### Analytical: DONE

* implement analytical brute-force solution: {YES\_DONE}

* test analytical solution: hard assert convergence tests (both need to converge, towards same limit): {YES\_DONE}

* insteafd of fixed n\_max, use "abbruchskriterium". terminate if |e\[-2] - e\[-1]| > 1e-8: {YES\_DONE}

* optimize analytical solution: {YES\_DONE}

* check convergence of analytical solution: compare with legacy elc: {YES\_DONE}
  * plot like this: <https://chat.icp.uni-stuttgart.de/icp/pl/pegsmbs8opbfzrchp9fhj5zs4y>: {YES\_DONE}

##### Neutral: DONE

###### Metallic ("delta\_mid\_bot = -1.0"): DONE

* create energy-distance plot: {YES\_DONE}
  * Dipol in kleinen Schritten von der Wand weg bewegen und immer die Energie messen, dass die Kurve schön glatt ist. Insbesondere über den "Sprung", wenn man for der near in die far-formula springt {YES\_DONE}

  * speed up ana solution: {YES\_DONE}

  * compare custom elcic: {YES\_DONE}

###### Non-Metallic: DONE

##### Non-Neutral

note: for single plate, there may not be a correct "correction term"

###### Metallic ("delta\_mid\_bot = -1.0"): DONE

###### Non-Metallic: NO, i dont have to do it. doesnt work

## Dual Plates

#### Analytical:

#### Neutral

##### Non-Neutral

* **Test:**
  * use neutral system: {YES\_DONE}
    * do Non-periodic slab with two metallic plates. (anaSolution in Tyagi et al. *JCP* 129, 2008). <https://doi.org/10.1063/1.3021064>: {YES\_DONE}
    * do mixed plates (if theres an anaSolution): {YES\_DONE}
    * do non metallic plate (if theres an anaSolution): {YES\_DONE}

  * create convergence / contribution plots (also good for debugging)
    * cleanup elcic.py: {YES\_DONE}
    * add get\_contribs(): {YES\_DONE}
    * plotting code: {YES\_DONE}
    * try error with espresso.ELC: {YES\_DONE} - error also plateaus at 1e-6
    * dont plot e\_l0 - instead do e\_corr, e\_3d, etc: {YES\_DONE}

  * find accurate anaEnergy (compare against legacyELC, must: err < 1e-6)
    * elc\_vs\_analytic.py: err 1e-6: {YES\_DONE}
    * anaEnergy (own impl): err 1e-6: {YES\_DONE}

  * implement elcic.py (with contribs): {YES\_DONE}

  * create contribution plots: {YES\_DONE}

  * compare different parameter sets
    * refac to allow different parameters
      * no more box\_l, independend lx,ly,lz: {YES\_DONE}
      * particles arent always (0, 0, z), (0, 0, z+d): {YES\_DONE}

#### Single Interface (Large Box - No Periodicity)

* single plate, metallic: {YES\_DONE}

* create analytical solution ("brute force"-Weg geht und Periodische Images)
  * check if it converges (for n\_itertations -> inf): {YES\_DONE}

* improve analytical (unti it fits legacy elc): {YES\_DONE}
  * test for varying neutral single-plate systems (box\_l, charges, ...): {YES\_DONE}

* add params label to plot: {YES\_DONE}

* improve analytical until it converges: {YES\_DONE}

* improve analytical until it matched legacy elc : NAJA
  * test for even contributions
  * test for varying neutral single-plate systems (box\_l, charges, ...):

* rigid tests

* clean up (pass params dict, not individual args)

* improve elcic.py until it fits ana/legacy

  * do non-neutral systems (if theres an anaSolution)
    * note: non-metallic and non-neutral might be impossible
    * eps\_top = eps\_bottom: easy ana solution
    * eps\_top != eps\_bottom: tricky, is kinda like single plate

* two plates

# vv UNORGANIZED vv

#### Full Periodic System

* **Test:** Periodic slab with two plates. Validate against a "brute force" direct summation.
  * do varying plate-combinations
  * do non-neutral systems
    * note: non-metallic and non-neutral might be impossible
* **Implement:** Integrate `elcic.py` into periodic elc.py to pass tests.

#### Parameter Sweep & Validation

* **Test:** Re-test all original ELC tests/plots; add tests for new parameters $\epsilon_{top}, \epsilon_{bottom}, \epsilon_{mid}$, etc.
* **Implement:** Update `elcic.py` to pass tests.

### Phase 2: Force Implementation & Testing

* **Procedure:** Follow the exact same workflow as the Energy implementation

### REFACTOR PYTHON CODEBASE

* check lz+gap\_size or lz-gap\_size consistency
  """

