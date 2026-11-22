# PerturbNMF

PerturbNMF runs consensus non-negative matrix factorization (cNMF) on Perturb-seq and other
single-cell CRISPR screen data. It infers gene programs, scores them, tests which perturbations
change them and helps you interpret the results. It is built around three stages:

- **Inference.** Run cNMF on the CPU (sk-cNMF) or on a GPU (torch-cNMF), across many values of K
  in a single run.
- **Evaluation.** Nine metrics for choosing K and checking program quality, plus calibrated
  perturbation tests (U-test, conditional randomization test, matched-cell DE).
- **Interpretation.** K-selection and per-program plots, an Excel summary, HTML reports, and
  LLM-assisted program annotation.

It runs on a single machine. SLURM runner scripts are included for clusters.

```{figure} _static/perturbnmf_overview.svg
:alt: PerturbNMF overview: input data, inference, evaluation and interpretation
:target: _static/perturbnmf_overview.pdf
:width: 100%
:class: only-light
:figclass: only-light

Overview of PerturbNMF. Click the figure for the full-resolution PDF.
```

```{figure} _static/perturbnmf_overview_dark.svg
:alt: PerturbNMF overview: input data, inference, evaluation and interpretation
:target: _static/perturbnmf_overview.pdf
:width: 100%
:class: only-dark
:figclass: only-dark

Overview of PerturbNMF. Click the figure for the full-resolution PDF.
```

## Installation

PerturbNMF is a set of scripts and notebooks. Clone the repository and create the conda
environment for the stage you want to run:

```bash
git clone https://github.com/EngreitzLab/PerturbNMF.git
cd PerturbNMF
conda env create -f src/Stage1_Inference/torch-cNMF/environment.yml
```

See {doc}`installation` for the other environments and the Stage 2 reference data.

## Documentation

New to PerturbNMF? {doc}`quickstart` runs Stage 1 on PBMC3k in a few minutes. The
{doc}`tutorials/index` go through each stage step by step, the {doc}`guide/index` covers the
{doc}`input data format <guide/data_format>`, {doc}`how to choose K <guide/choosing_k>` and every parameter and output, and the {doc}`explorer` shows programs from the paper datasets.

## Citation

If you use PerturbNMF, cite the concept DOI
[10.5281/zenodo.21878267](https://doi.org/10.5281/zenodo.21878267). See {doc}`citation`.

## Featured publications

**Studies using PerturbNMF**

- Nguyen & King *et al.*, 2026. [“Mapping the regulatory wiring of endothelial cell differentiation reveals a critical role for Hippo mechanosignaling.”](https://www.biorxiv.org/content/10.64898/2026.11.04.763740v1) *bioRxiv*.
- Ma, Conley & Chen *et al.*, 2024. [“Interpreting variants associated with congenital heart disease with a regulatory map of the human fetal heart.”](https://www.medrxiv.org/content/10.1101/2024.11.20.24317557v3) *medRxiv*.

**Previous studies**

- Schnitzler & Kang *et al.*, 2024. [“Convergence of coronary artery disease genes onto endothelial cell programs.”](https://www.nature.com/articles/s41586-024-07022-x) *Nature*.

```{toctree}
:hidden:

quickstart
installation
tutorials/index
guide/index
explorer
changelog
citation
```
