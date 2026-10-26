# Perturbation Association Calibration

This folder contains three statistical tests for evaluating perturbation effects on gene programs identified by cNMF. Each test provides different statistical frameworks for assessing whether perturbations significantly affect gene program activity.

> **⚠️ Conda environments differ across Stage 2 substages — do NOT use the same env for everything.**
>
> | Substage | Path | Required conda env |
> |---|---|---|
> | 2a Evaluation Metrics | `src/Stage2_Evaluation/A_Metrics/` | `NMF_Benchmarking` |
> | **2b U-test calibration** (this folder) | `src/Stage2_Evaluation/B_Calibration/Slurm_version/U-test_perturbation_calibration/` | `NMF_Benchmarking` |
> | **2b CRT calibration** (this folder) | `src/Stage2_Evaluation/B_Calibration/Slurm_version/CRT/` | `NMF_Benchmarking` |
> | **2b Matched Cell DE (R)** (this folder) | `src/Stage2_Evaluation/B_Calibration/Slurm_version/Matched_cell_programDE/` | `gene_propagation` |
>
> **U-test** and **CRT** calibration are written in Python and share the **same `NMF_Benchmarking` env as the Stage 2a evaluation metrics** — no separate install is needed if you already have that env.
>
> **Matched Cell DE** is written in **R** and requires a **separate `gene_propagation` env** with additional packages installed (it ships its own `environment.yml`).
>
> Running a script under the wrong env will fail with `ModuleNotFoundError` (Python) or `package not found` (R). Always activate the env shown above before launching a `.sh` from that folder.

## Overview of Tests

### 1. U-test Perturbation Calibration
**Location:** `JupterNote_Version/U-test_perturbation_calibration.ipynb` and `Slurm_version/U-test_perturbation_calibration/`

**Description:** Computes perturbation association tests on real and fake (calibration) data using Mann-Whitney U-test to evaluate the statistical properties of perturbation detection methods.

**Method:**
- Uses non-parametric Mann-Whitney U-test to compare program usage between perturbed and control cells
- Tests both real perturbations and synthetic calibration data (non-targeting guides)
- Evaluates false positive rates through QQ plots comparing observed vs. expected p-value distributions

**Key Features:**
- Non-parametric approach (no distributional assumptions)
- Calibration analysis to assess statistical validity
- Suitable for initial exploratory analysis

---

### 2. CRT (Conditional Randomization Test)
**Location:** `JupterNote_Version/CRT.ipynb` and `Slurm_version/CRT/` (package: `src/CRT/`; full docs in `Slurm_version/CRT/README.md`)

**Description:** Uses SCEPTRE-style conditional randomization tests (CRT) to test perturbation effects on gene programs while accounting for technical covariates. Runs separately per condition (`--categorical_key`). The notebook mirrors `CRT.py` step by step for one condition.

**Method:**
- Per-program OLS of the program usage share (`--normalization row_normalize`, default) or CLR (`clr`) on the guide indicator + covariates (one-hot / z-scored, with an intercept)
- `--moi high` (default): CRT — treated sets resampled Bernoulli(p) from a per-gene unpenalized logistic propensity model; controls = all other cells (or `--control_group ntc_cells|matched`)
- `--moi low`: permutation test against control-guide cells within covariate strata
- Two-stage resampling: `--number_permutations` stage-1 rounds (default 4999), plus `--extra_resampled_rounds` (default 49999) for programs with stage-1 p < 0.05; skew-normal tail p-value for those programs (`--pvalue_method skew_normal`, default)
- NTC null: fake genes of `--number_guide` non-targeting guides, frequency-matched to real genes (`--null_method frequency`) or random, run through the same test

**Key Features:**
- Robust to covariates through the conditional randomization framework
- Calibration checked on a QQ plot of real vs NTC-null p-values
- Skew-normal fit resolves p-values below the empirical floor 1/(1 + rounds)
- More computationally intensive (parallelized over genes with `--n_jobs`) but statistically rigorous

**Parameters (main ones; see `Slurm_version/CRT/README.md` for all):**
- `--number_permutations`: Stage-1 resampling rounds per gene (default: 4999)
- `--extra_resampled_rounds`: Stage-2 rounds for programs with stage-1 p < 0.05 (default: 49999; 0 disables)
- `--number_guide`: NTC guides per fake (null) gene (default: 6)
- `--moi`: `high` (CRT) or `low` (permutation) (default: high)
- `--normalization`: `row_normalize` or `clr` (default: row_normalize)
- `--pvalue_method`: `skew_normal` or `empirical` (default: skew_normal)
- `--covariates`: Covariate keys in .obs to include as-is
- `--log_covariates`: Covariate keys in .obs to log1p-transform before inclusion
- `--guide_annotation_key`: Name(s) of non-targeting guide target labels (default: "non-targeting")
- `--FDR_method`: FDR correction method: "BH" or "StoreyQ" (default: "BH")
- `--n_jobs`: Parallel workers over genes (default: `$SLURM_CPUS_PER_TASK` or 1)
- `--save_dir`: Base directory for the `{K}_{thresh}/` result folders (defaults to `<out_dir>/<run_name>/Evaluation/`)

---

### 3. Matched Cell ProgramDE
**Location:** `Slurm_version/Matched_cell_programDE/`

**Description:** Performs statistical testing using propensity score matching and regression with G-computation to identify significant perturbation effects on gene programs. Written by Tony Zeng.

**Method:**

#### Pipeline Steps:

1. **Cell Matching with Propensity Score Model**
   ```
   logit(πc) = log(Pr(Tc = 1|Xc) / (1 - Pr(Tc = 1|Xc))) = γ⊤Xc
   ```
   Select matched cells (perturbed vs unperturbed) to reduce confounding.

2. **Program Score Transformation**
   ```
   Ỹck = log(Pck + ψk)
   ```
   Apply log transformation with adaptive pseudocount to program scores.

3. **Regression with G-computation**
   ```
   Ỹck = β0 + βgTc + β⊤XcXc + ϵck
   ```
   - `Tc`: treatment indicator (1 for perturbed, 0 for unperturbed)
   - G-computation estimates `beta_effect` by computing difference in predicted outcomes:
     - `Ŷ_treatment`: predicted value when setting Tc = 1 for all cells
     - `Ŷ_control`: predicted value when setting Tc = 0 for all cells
     - `beta_effect = mean(Ŷ_treatment) - mean(Ŷ_control)`

4. **Variance Estimation**
   Use delta method to calculate variance and standard deviation of effect size.

5. **Statistical Testing**
   Compare `(beta_effect, std)` to normal distribution using two-sided test to obtain p-values.

**Null Model Testing:**
For null model validation, group 6 guides into 1 pseudo gene and follow the same process to generate null p-value distributions.

**Key Features:**
- Causal inference framework using propensity score matching
- Accounts for confounding covariates through matching
- G-computation provides interpretable effect sizes
- Variance estimation through delta method
- Written in R with detailed documentation

---

## Output Organization

### U-test
```
Evaluation/
├── config_{job_id}.yml
└── {k}_{sel_thresh}/
    ├── {k}_perturbation_association_results_{sample}.txt   (one per sample)
    └── {k}_fake_perturbation_association_results.txt
```
Plus visualization outputs at `<out_dir>/<run_name>/`:
- `perturbation_association_calibration.png`
- `perturbation_association_qqplot_overlap.png`

### CRT
```
Evaluation/
├── config_{job_id}.yml
└── {k}_{sel_thresh}/
    ├── {k}_CRT_{covar_tag}_{condition}.txt                    (real results, one per condition)
    ├── {k}_CRT_fake_{covar_tag}_{condition}.txt               (NTC null)
    ├── {k}_CRT_{covar_tag}_{condition}_qq_empirical.pdf       (QQ plots, real vs NTC null)
    └── {k}_CRT_{covar_tag}_{condition}_qq_skew_normal.pdf
```

### Matched Cell ProgramDE
```
Evaluation/
└── {k}_{sel_thresh}/Matched_cell_programDE/
    ├── {condition}_gene_results.csv            (per condition)
    ├── {condition}_null_results.csv            (per condition)
    └── logs/
```

## Choosing a Test

**Use U-test when:**
- Performing initial exploratory analysis
- Need fast, non-parametric testing
- Want to assess overall calibration properties

**Use CRT when:**
- Need robust covariate adjustment
- Want permutation-based statistical rigor
- Have sufficient computational resources for permutation tests

**Use Matched Cell ProgramDE when:**
- Strong confounding is suspected
- Want interpretable causal effect sizes
- Need propensity score matching framework
- Prefer regression-based approach

## Notes

- All tests require muon/scanpy objects with cNMF program usage scores
- Guide assignment and metadata must be properly formatted
- Non-targeting control guides are required for null model calibration
- Environment files are provided for each test where applicable
