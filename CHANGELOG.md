# Changelog

All notable changes to PerturbNMF are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked ⚠️ change pipeline output — re-run affected analyses.

## [Unreleased]

### Added
- CRT `--resampling fixed_count` (`resampling="fixed_count"` in
  `run_all_genes_union_crt`, `run_one_gene_union_crt` and the NTC functions): null
  treated sets keep the observed treated count within each categorical-covariate
  stratum. Exact for categorical-only covariates; approximate (Pareto sampling on the
  propensity) with continuous covariates. Default stays `bernoulli`.
- CRT `--outcome usage` (`outcome="usage"` in `prepare_crt_inputs`): Y is the per-cell
  program usage share (usage row-normalized, no log, no floor) instead of the CLR of
  floored usage; the effect is written as `usage_share_diff` (difference in mean usage
  share) instead of `log2FC`. Default stays `clr`. Recommended with
  `--resampling fixed_count`.
- CRT `--matched_ntc_null`: cell-count-matched NTC pseudo-targets as a calibration
  diagnostic for few-cell targets (`{K}_CRT_matched_null_*.txt`).
- CRT NTC null now also carries the skew-normal p-value (`p-value`, `adj_pval` in
  `{K}_CRT_fake_*.txt`), plus a `_skew.png` QQ plot and a `_skew` NTC-significance
  summary on that scale.

### Fixed
- ⚠️ Two-sided skew-normal and empirical CRT p-values are clipped at 1 (could reach
  ~1.04). Only p-values that were > 1 change.

## [0.1.1] - 2026-08-10

### Added
- Zenodo archiving of GitHub releases. `.zenodo.json` and `CITATION.cff` supply
  the deposition metadata; the README carries a concept-DOI badge and a Citation
  section. Bump `version` and `date-released` in `CITATION.cff` with each release.

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

[Unreleased]: https://github.com/EngreitzLab/PerturbNMF/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/EngreitzLab/PerturbNMF/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EngreitzLab/PerturbNMF/releases/tag/v0.1.0
