# Action Tree

fix near energy, "a/b\_zshift\_" contains error spike at pos.z=lz/2

* debug/fix by hand: NO, super expensive and slow
* using gemini chatbot (prompts, pasting code, etc): NO, didnt work in 20 tries &#x20;
* using ai agent: MEH, slight improvement
* IDEA: why does LLM: "convert elc.cpp to python" not work?
  * can i do it contrib-wise (e.g. "convert e\_near from elc-cpp to python?)
  * insert contrib print statements in elc.cpp

fix Far-Field Isolation (Large Gap) (e\_far = 1e-5) so my accuracy has to be like 1e-6 / 1e-8.

* Use `gap_size >> λ` (e.g., gap = 30, lz = 10) so near-field image charges are very distant.
* Expected:`e_near` → 0 (images far away), energy dominated by `e_far`. If total energy is wrong here, the bug is in `_get_far_field_energy`.

# Problems

* legacy elcic (and analytical, if available) only retunr E\_total.  it doesnt help with finding errors in individual contribs, etc
* huge parameter range, even if it works for most params, theres always regions with errors

