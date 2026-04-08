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

anaSolution in Tyagi et al. *JCP* 129, 2008: <https://doi.org/10.1063/1.3021064>

* make it converge
* make it fit legacy ELC in dipole shifting

#### Neutral

* do two metallic plates.
* do non metallic plate (if theres an anaSolution)
* do mixed plates (if theres an anaSolution)

#### Non-Neutral

###### Non-Metallic: NO, dont do it, doesnt work

* eps\_top = eps\_bottom: easy ana solution
* eps\_top != eps\_bottom: tricky, is kinda like single plate

# vv UNORGANIZED vv

do i merge elc.py and elcic.py? is elcic.py a superset of elc.py?

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

### Phase 3: Refactor python codebase

* check lz+gap\_size or lz-gap\_size consistency

