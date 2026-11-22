# Recommended workflow

A typical path from a new dataset to a final set of programs. Each step links to the page with the details.

1. **Prepare the input.** Raw counts plus guide information, laid out as in {doc}`data_format`.
2. **Use as many highly variable genes as you can afford.** More genes in the factorization means more genes to
   test for enrichment, and usually more enriched terms per program. Raise `--numhvgenes` as far as your time and
   memory allow ({doc}`torch_cnmf`).
3. **On large datasets, converge further.** Use a lower convergence tolerance and more iterations, so the replicates
   agree and the consensus programs are stable.
4. **Sweep a wide range of K first.** A broad, coarse sweep (for example K = 10 to 300) shows the overall trend of
   the stability and biology metrics ({doc}`../tutorials/k_selection`).
5. **Then sample densely around the best range.** If the broad sweep points to a region, rerun with closely spaced
   K values there.
6. **Check perturbation calibration.** Look at the calibration plots to choose how p-values are estimated (U-test,
   CRT or matched-cell DE); see {doc}`calibration`.
7. **Pick K from all the evidence.** Weigh stability, the biology metrics and how interpretable the programs look;
   see {doc}`choosing_k`.
8. **Pick a density threshold for that K.** Compare a few thresholds; see {doc}`density_threshold`.
9. **Make the final plots for the chosen K.** Program and perturbed-gene reports, the Excel summary and, if wanted,
   program annotation ({doc}`plotting`, {doc}`summarization`, {doc}`annotation`).

```{note}
Program annotation calls a paid LLM API, so every run costs money. Try it on a few programs before annotating all
of them.
```
