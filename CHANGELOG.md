# Changelog

All notable changes to PerturbNMF are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked ⚠️ change pipeline output — re-run affected analyses.

## [Unreleased]

## [0.1.0] - 2026-08-06

First release.

### Added
- **Stage 1 Inference** — cNMF via sk-cNMF (CPU, scikit-learn) or torch-cNMF
  (GPU, PyTorch), with batch, minibatch, and parallel SLURM runners.
  Outputs `cNMF_{K}_{thresh}.h5mu` with program scores and gene loadings.
- **Stage 2 Metrics** — 9 evaluation metrics: categorical association,
  perturbation sensitivity, motif enrichment, GWAS/OpenTargets trait
  enrichment, GO and geneset enrichment, explained variance, reconstruction
  error, stability.
- **Stage 2 Calibration** — three perturbation-calibration methods: U-test
  (fast, non-parametric), CRT (permutation-based, covariate-adjusted), and
  matched-cell DE (R, `programDE`). Null p-values are cached so re-runs skip
  recomputation.
- **Stage 3 Interpretation** — K-selection plots, per-program QC plots,
  perturbation plots, Excel summarization, and LLM-driven program annotation
  with PubTator3 literature mining.
- Four Claude Code skills under `.claude/skills/` for guided pipeline
  execution, `.h5mu` inspection, test-suite runs, and parameter-drift checks.
- MIT license.

[Unreleased]: https://github.com/EngreitzLab/PerturbNMF/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EngreitzLab/PerturbNMF/releases/tag/v0.1.0
