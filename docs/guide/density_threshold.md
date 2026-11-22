# Choosing the density threshold

cNMF runs NMF many times and clusters the resulting programs into consensus programs. Before clustering, it removes
outlier programs whose mean distance to their nearest neighbors is above the **density threshold**
(`--sel_thresh` in Stage 1, `--sel_threshs` in Stages 2 and 3). A good threshold removes only the few, isolated
outliers.

## Rule

Look at the local density histogram for your K and ask:

1. Does the threshold cut off only a separate tail of low-density outliers?
2. Does it remove only a small fraction of programs?

If both are true, the threshold is a good choice.

## Reading the plots

**Per K** (Stage 1 diagnostic plot):

- Left: clustermap of the pairwise similarity between the consensus programs.
- Right: histogram of each program's mean distance to its nearest neighbors, with the threshold as a dashed line.

**Per threshold, across all K:**

- Top: stacked bars of program replicates kept (green) and filtered out (grey) at each K.
- Bottom: the percentage kept at each K.

````{list-table}
:widths: 50 50

* - ```{figure} ../_static/guide/density_survival_thresh2.0.png
    :alt: At threshold 2.0 every replicate survives at every K

    Threshold 2.0: nothing is filtered.
    ```
  - ```{figure} ../_static/guide/density_survival_thresh0.2.png
    :alt: At threshold 0.2 the share of surviving replicates drops as K grows

    Threshold 0.2: the share filtered grows with K.
    ```
````

## Examples

All three use threshold 0.2.

### Good: iPSC-derived endothelial cells

100k cells × 17,538 genes, 20 NMF replicates. The threshold cuts off a separate outlier tail, and few programs are lost.

````{list-table}
:widths: 50 50

* - ```{figure} ../_static/guide/density_ex1_k30.png
    :alt: Density histogram at K = 30 with a small outlier tail beyond the threshold

    K = 30: 51 of 600 filtered (8%). Good.
    ```
  - ```{figure} ../_static/guide/density_ex1_k300.png
    :alt: Density histogram at K = 300 with a small outlier tail beyond the threshold

    K = 300: 1,021 of 6,000 filtered (17%). Good.
    ```
````

### Bad: iPSC-derived cardiomyocytes

1M cells × 600 genes, 10 NMF replicates. The threshold cuts through the main body of the distribution, so most
programs are removed.

````{list-table}
:widths: 50 50

* - ```{figure} ../_static/guide/density_ex2_k30.png
    :alt: Density histogram at K = 30 where the threshold cuts through the main peak

    K = 30: 151 of 300 filtered (50%). Too many.
    ```
  - ```{figure} ../_static/guide/density_ex2_k250.png
    :alt: Density histogram at K = 250 where the threshold cuts through the main peak

    K = 250: 2,115 of 2,500 filtered (85%). Too many.
    ```
````

### Depends on K: iPSC-derived pancreatic cells

100k cells × 2,000 genes, 100 NMF replicates. The same threshold works at low K but not at high K.

````{list-table}
:widths: 50 50

* - ```{figure} ../_static/guide/density_ex3_k25.png
    :alt: Density histogram at K = 25 with a small outlier tail

    K = 25: 435 of 2,500 filtered (17%). Good.
    ```
  - ```{figure} ../_static/guide/density_ex3_k300.png
    :alt: Density histogram at K = 300 where most programs lie beyond the threshold

    K = 300: 25,828 of 30,000 filtered (86%). Too many.
    ```
````

## Pitfalls

- If the threshold removes more than half of the programs, run more NMF replicates (`--numiter`) or use a larger
  threshold.
