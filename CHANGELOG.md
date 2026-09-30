# Changelog

All notable changes to PerturbNMF are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked ⚠️ change pipeline output — re-run affected analyses.

## [Unreleased]

### Changed
- The pipeline now uses the term "condition" throughout. In the Excel summary,
  `--Sample` is renamed `--Conditions` (`--Sample` still works as a deprecated
  alias). `--categorical_key` help text now describes it as the `.obs` column
  holding each cell's condition label.
- The Excel summary functions in `Compile_excel_sheet.py` (`Compile_Perturbation_sheet`,
  `Compile_Target_Summary_sheet`, `Compile_Summary_sheet`, `load_simple_sheets`, …)
  take `conditions=` instead of `Sample=`, matching `load_perturbation_data`.
  The `# programs <condition>` columns are now integers.
- Condition labels no longer default to one study's values: when `--Conditions`
  is omitted (plotting, K-selection, Excel summary), labels are read from
  `obs[categorical_key]` in the h5mu.
- SLURM runner scripts are templates: set `PIPELINE_ROOT` and fill in the
  `<partition>` / `<your_email>` placeholders. Python entry points resolve the
  repo from their own location instead of a hardcoded checkout path.
- Annotation defaults are cell-type neutral; the PubMed keyword is optional, and
  Vertex AI project/bucket come from env vars (`VERTEX_PROJECT_ID`, `VERTEX_BUCKET`).

### Added
- `tools/check_no_lab_specific_content.py` guard (CI + pre-commit) that blocks
  lab/project-specific content in this public repo.

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
