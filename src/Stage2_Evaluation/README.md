# Stage 2 — Evaluation

After Stage 1 produces consensus cNMF programs, Stage 2 evaluates them along
two complementary axes:

| Substage | What it answers | Folder |
|----------|------------------|--------|
| **2a. Metrics** | Are the programs biologically meaningful and technically reproducible? | [`A_Metrics/`](A_Metrics/README.md) |
| **2b. Calibration** | Are perturbation-association statistics well-calibrated (controlled false-positive rate)? | [`B_Calibration/`](B_Calibration/README.md) |

## 2a. Metrics (`A_Metrics/`)

Runs 9 evaluation criteria per program:

- **Categorical association** — does the program differ across batches / conditions?
- **Perturbation sensitivity** — does it shift under direct perturbation of its top genes?
- **Motif enrichment** — are top genes co-regulated by shared TFs (HOCOMOCO)?
- **Trait enrichment** — Fisher's test vs OpenTargets GWAS L2G
- **GO + gene-set enrichment** — GSEA against GO and MSigDB/Enrichr
- **Explained variance / Reconstruction error / Stability** — overall fit and reproducibility

See [`A_Metrics/README.md`](A_Metrics/README.md) for the criterion table and CLI usage.

## 2b. Calibration (`B_Calibration/`)

Three statistical frameworks for perturbation–program association testing,
each with its own statistical assumptions and conda environment:

- **U-test** (non-parametric Mann–Whitney) — `NMF_Benchmarking`
- **CRT** (Conditional Randomization Test) — `programDE`
- **Matched-cell DE** (R, paired perturbed/control cells) — `gene_propagation`

Each method runs both real and "fake" (non-targeting) tests to produce QQ plots
that diagnose calibration. See [`B_Calibration/README.md`](B_Calibration/README.md).

## Shared resources

[`Resources/`](Resources/) holds reference files used across substages:
HOCOMOCO motif file, OpenTargets L2G GWAS table, hg38 genome FASTA.

## Conda environments

> ⚠️ Different substages need different envs — don't reuse one for everything.

| Substage | Env |
|----------|-----|
| 2a Metrics | `NMF_Benchmarking` |
| 2b U-test | `NMF_Benchmarking` |
| 2b CRT | `programDE` |
| 2b Matched-cell DE | `gene_propagation` |

Activate the right env before launching any `.sh` in that folder; running under
the wrong env will fail with `ModuleNotFoundError` (Python) or
`package not found` (R).
