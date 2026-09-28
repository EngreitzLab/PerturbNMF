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
- **Motif enrichment** — are the top genes' promoters / enhancers enriched for a TF motif (FIMO with
  MotifCompendium-Database-Human clusters by default, or HOCOMOCO v11; and/or ChromBPNet Fi-NeMo hits; grouped by motif family; Welch t-test as in Schnitzler et al. 2024, or loading
  correlation)? Also nominates candidate TFs (enriched + expressed / knockdown regulates the program).
  Driver `A_Metrics/Slurm_Version/run_motif_enrichment.py` or `cNMF_evaluation_pipeline.py --Perform_motif`;
  method, defaults, resource downloads and validation in [`A_Metrics/README.md#motif-enrichment`](A_Metrics/README.md#motif-enrichment).
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
HOCOMOCO motif file, OpenTargets L2G GWAS table, hg38 genome FASTA. Motif enrichment scans
MotifCompendium-Database-Human (kundajelab; its metadata / TF lists are bundled in `A_Metrics/src/motif_databases/`)
or HOCOMOCO v11 (`--motif_db hocomoco_v11`); pass the MEME file, genome FASTA and GTF as flags or via
`$PERTURBNMF_MOTIFCOMPENDIUM_MEME`, `$PERTURBNMF_GENOME_FASTA`, `$PERTURBNMF_GENE_ANNOTATION`. Enhancer links and
Fi-NeMo files are per cell type (see `A_Metrics/src/find_regulatory_resources.py`).

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
