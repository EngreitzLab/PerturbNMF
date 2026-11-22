# Choosing K

K is the number of programs. The K-selection plot ({doc}`plotting`) shows how a set of metrics changes across the
K values you ran.

```{figure} ../_static/guide/k_selection_example.png
:alt: K-selection plot with eight panels; the curves level off around K = 50
:width: 100%

K-selection plot for iPSC-derived endothelial cells (100k cells × 17,538 genes, 20 NMF replicates). The dashed
line marks the chosen K = 50.
```

## Rule

When a curve levels off, adding programs no longer captures new signal. Pick the K at the turning point where the
curves plateau: beyond it, extra programs bring diminishing returns. In the example above that elbow is at K = 50.

## Reading the panels

| Panel | Metric | What to look for |
|---|---|---|
| A | Program stability | High and flat at the chosen K. |
| B, C | Reconstruction error, explained variance | Both keep improving as K grows. Look for the knee, not the best value. |
| D, E, F | Unique GO terms, gene sets, traits | How much distinct biology the programs capture. A plateau means no new themes are appearing. |
| G, H | Unique regulators, across all samples and per sample | How many regulators significantly affect at least one program. Prefer K values with many regulators. |

## Pitfalls

- Don't maximize explained variance (panel C): it keeps rising with K.
- Don't maximize stability (panel A): it is often highest at a trivially small K.
- If every panel is still rising at your largest K, extend the sweep upward before deciding.
- Trust the panels that agree. Lean on D, E and G/H more than on A, B and C.

Next: {doc}`density_threshold`.
