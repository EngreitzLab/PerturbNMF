# How to choose K (number of programs)

This directory contains an example K-selection diagnostic plot
(`K-selection_Example.png`) and the corresponding gene-/program-level PDF
reports. Use the plot to decide how many cNMF programs (K) to retain for
downstream analysis.

## The guiding principle

> **The saturation of the curves serves as a proxy for whether all signal has
> been captured. Choose the K at the *turning point*, where the curves
> plateau** — adding more programs after that yields diminishing returns and
> mostly fragments existing signal into smaller pieces.

The red dashed line in `K-selection_Example.png` marks the chosen K = 50 for
that run, identified at the elbow shared across most panels.

## What each panel measures

The K-selection plot has 8 panels (A–H). Each curve is a different diagnostic
evaluated across a sweep of K values.

| Panel | Y-axis | What it tells you | Direction at the right K |
|-------|--------|-------------------|--------------------------|
| **A** Program stability | Silhouette score of consensus clusters | How reproducibly the same programs appear across NMF replicates | Higher is better; falls quickly then plateaus |
| **B** Reconstruction error | Frobenius prediction error on the held-out matrix | How well the model fits the data | Lower is better; decreases then flattens |
| **C** Explained variance | Total variance explained by the K programs combined | How much of the data signal the model captures | Monotonically increases; look for the knee, not the peak |
| **D** GO term enrichment | Number of unique GO terms significantly enriched across programs | Whether programs are tagging distinct biology | Plateau = no new biological themes being discovered |
| **E** Gene set enrichment | Number of unique gene sets enriched (MSigDB, etc.) | Same idea as D, on a different ontology | Plateau = no new pathways |
| **F** Trait enrichment | Number of unique GWAS traits enriched | Whether programs are linked to additional traits/diseases | Plateau = no new trait associations |
| **G** Unique regulators for all samples | Distinct TFs / regulators implied by programs (pooled) | How much regulatory vocabulary the model recovers | Plateau = no new regulators identified |
| **H** Unique regulators per sample | Same as G, broken down per sample/condition | Per-condition regulator diversity | Plateaus may differ across samples |

Each panel is independent evidence. **Trust the panels that agree.** When the
turning point lines up across A–F, pick that K. If panels disagree, lean on
panels A, D, E (stability + biological signal) over B, C (which always trend
monotonically and can mislead).

## How to actually pick K from your own run

1. **Run the inference pipeline** for a broad sweep of K values
   (e.g. `K=[10, 20, 30, 50, 80, 100, 150, 200, 250, 300]`). Use the
   `torch_cnmf_inference_pipeline.py` runner — set `--run_factorize --run_refit`.

2. **Generate the K-selection plot** using the interpretation pipeline.
   See `examples/Interpretation/` for the expected output format.

3. **Look at each panel** for the elbow:
   - Find the point on each curve where the slope visibly drops.
   - Mark that K mentally on each panel.

4. **Pick the K where most panels agree** — typically the elbow on Panel A
   (stability) and Panel D (GO terms) is the most reliable signal.

5. **Sanity-check with the density-filtering plot**
   (`<run>/Inference/Inference.density_filtering.dt_<X>.png`) — at very large
   K, most program replicates get filtered out as outliers, indicating the
   model is over-fragmenting. If survival rate has collapsed at your chosen K,
   pick a smaller K.

6. **Look at the example K = 50 in this folder** to see what a well-supported
   choice looks like:
   - Panel A: stability has fallen and stabilized at ~0.75
   - Panels D–F: clear knee — most biological signal already captured
   - Panel C: explained variance is still climbing, but the rate of gain is
     dropping (knee)
   - Larger K (100, 150, 200, 250, 300) adds very little new signal in D–F
     while continuing to inflate the program count

## Common pitfalls

- **Don't pick the K that maximizes explained variance.** Panel C keeps
  climbing forever — picking the highest K just means you're chasing noise.
- **Don't pick the K that maximizes stability.** Panel A is often *highest at
  small K* simply because there are fewer programs to cluster; that doesn't
  mean those few programs capture all the biology.
- **Watch out for "all panels still rising"** at the largest K you ran. That
  means the sweep did not reach saturation — extend the range upward
  (e.g. add `K=[400, 500]`) before deciding.
- **Stability dips at one specific K** (a single bad point on Panel A) are
  usually NMF-replicate noise. Look at the shape of the curve, not one point.

## Files in this directory

| File | What it is |
|------|------------|
| `K-selection_Example.png` | Example K-selection diagnostic (8 panels, red line at K=50) |
| `Program_PDF_Report_Example.pdf` | Per-program annotation report at the chosen K |
| `Gene_PDF_Report_Example.pdf` | Per-gene cross-program report at the chosen K |
