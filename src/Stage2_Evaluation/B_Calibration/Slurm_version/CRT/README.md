# CRT

## Overview
A pipeline for testing differential effect (DE) of gene targets on gene programs. This pipeline implements a Conditional Randomization Test (CRT) to assess the statistical significance of perturbations.

The core CRT logic is **vendored locally** in the `CRT` package at `src/Stage2_Evaluation/B_Calibration/src/CRT/` (adapted from the SCEPTRE-style tests in https://github.com/edtnguyen/programDE, written by Tri Nguyen). `CRT.py` imports it directly by appending `B_Calibration/src` to `sys.path` — no external/PyPI install of `src.sceptre` is required anymore.

## Setup (required)

CRT is written in Python and runs in the **`NMF_Benchmarking`** conda env — the **same env as the Stage 2a evaluation metrics and the U-test calibration**. If you already have that env, no separate install is needed. Otherwise create it from the sibling `environment.yml`, which contains every dependency the vendored `CRT` package needs (`numba`, `scipy`, `statsmodels`, `joblib`, `muon`, `scanpy`, `multipy`, ...):

```bash
conda env create -f environment.yml
conda activate NMF_Benchmarking
```

> **Note:** Earlier versions used a dedicated `programDE` env and installed the wrapper from GitHub (`pip install git+https://github.com/edtnguyen/programDE.git`), importing `src.sceptre`. Both steps are no longer needed — the package is now vendored under `src/CRT/` and imported locally, and CRT shares the `NMF_Benchmarking` env with the other Python evaluation metrics. `CRT.sh` / `CRT_parallel.sh` also export `PYTHONPATH=.../PerturbNMF/src` for good measure.

---

## Method overview

This pipeline tests **candidate regulator → program usage** effects using a per-program linear model, and assigns significance with a Conditional Randomization Test (CRT) followed by Storey q-value FDR control.

### Model

Let $i$ index cells and $k$ index programs. $Y_{i,k}$ denotes the CLR-transformed usage of program $k$ in cell $i$, $X_i$ is a binary indicator of whether a guide targeting the candidate regulator is present in cell $i$, and $C_{ij}$ indicates the covariate values for the given cell $i$. The coefficient $\beta_k$ captures the perturbation effect on program $k$, while each $\gamma_j$ captures the effect of covariate $j$. The fitted $\beta_k$ serves as the covariate-adjusted effect size of that regulator on the program.

**Compact form:**

```math
Y_{ik} = \beta_k X_i + \sum_j \gamma_j C_{ij} + \varepsilon_i
```

**Expanded form:**

```math
Y_{ik} = \beta_k X_i
+ \gamma_1 (\%\mathrm{mito})_i
+ \gamma_2 (\mathrm{replicate})_i
+ \gamma_3 (\mathrm{total\ counts})_i
+ \gamma_4 (\mathrm{guide\ UMI})_i
+ \gamma_5 (\mathrm{guide\ number})_i
+ \gamma_6 (\mathrm{gene\ number})_i
+ \varepsilon_i
```

Solve the system of equations above to obtain the effect sizes $\beta_k$ via OLS in Python.

### Conditional Randomization Test (CRT)

P-values were computed by comparing the observed effect size to a null distribution of resampled effect sizes, following the CRT framework. For each cell $i$, we first estimated the propensity score — the probability that the cell carries a guide targeting the candidate regulator given its covariates — using a logistic regression model fit on the same set of covariates $C_i$:

**Propensity model:**

```math
p_i = P(X_i = 1 \mid C_i) = \frac{1}{1 + e^{-t_i}},
```

where $t_i = \sum_j \gamma_j C_{ij}$ is the linear predictor from the propensity model. We then generated $B$ resampled guide assignments by drawing from the Bernoulli distribution defined by each cell's propensity score:

**Resampling step:**

```math
x_i^{(b)} \sim \mathrm{Bernoulli}(p_i), \qquad b = 1, 2, 3, \dots, B.
```

For each resample $b$, we refit the regression model with the resampled guide indicator to obtain a null effect size $\beta_k^{(b)}$:

**Null regression:**

```math
Y_{ik} = \beta_k^{(b)} x_i^{(b)} + \sum_j \gamma_j^{(b)} C_{ij} + \varepsilon_i.
```

The empirical two-sided p-value for program $k$ was then computed as the fraction of resampled effect sizes whose magnitude equaled or exceeded the observed effect size, with the standard $+1$ correction in both numerator and denominator to prevent p-values of zero:

**Empirical p-value:**

```math
P_k = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\!\left(\left|\beta_k^{(b)}\right| \ge \left|\beta_k^{(\mathrm{obs})}\right|\right)}{B + 1}.
```

To control the false discovery rate (FDR) across the full set of regulator–program tests, we applied the Storey q-value procedure. Regulator–program pairs with $q < 0.05$ were considered to be significant.

### Null calibration via NTC guide-group ensembles

To check that the CRT p-values are well calibrated, we generate a **negative-control null** from the non-targeting control (NTC) guides and compare it against the real target p-values on a QQ plot. This replaces the older leave-one-out-on-$\hat\beta$ scheme.

The null is built to look like real targets rather than being drawn arbitrarily:

1. **Frequency binning** (`build_ntc_group_inputs`). Compute each guide's prevalence $\mathrm{freq}(g)=\text{mean}(G[:,g]>0)$. Bin the *real* (targeting) guides into `n_bins` frequency quantiles, and record, for every real gene, the frequency-bin composition ("bin signature") of its guide set.

2. **Matched NTC groups** (`make_ntc_groups_ensemble` → `make_ntc_groups_matched_by_freq`). Randomly partition the NTC guides into synthetic "pseudo-gene" groups of size `--number_guide`, choosing guides so each group's frequency-bin composition matches a randomly drawn real-gene bin signature. This is repeated `n_ensemble` times (different seeds) to build an ensemble of null groupings. Because the NTC groups match real genes in both group size and per-guide prevalence, they form a fair null.

3. **Null p-values** (`crt_pvals_for_ntc_groups_ensemble`). Run the exact same union-CRT (§2.1–2.4) on each NTC pseudo-gene group across all programs, producing a null distribution of p-values.

4. **QQ diagnostic** (`qq_plot_real_vs_null`). Plot expected vs observed $-\log_{10}(p)$ for the real target p-values (purple) against the NTC null (blue). Well-calibrated null points track the $y=x$ diagonal; real points rising above it indicate genuine perturbation signal. One `{K}_CRT_{covariates}_{condition}.png` is written per (K, sel_thresh, condition).

> **Note:** the NTC group size is controlled by `--number_guide` — it is no longer hardcoded to 6, so it must match how many guides-per-gene your real targets use. The ensemble count (`n_ensemble`), bin count (`n_bins`), and seeds are currently set inside `CRT.py`/the package, not exposed as CLI flags.

---

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| out_dir | str | Directory containing cNMF output files |
| run_name | str | Name of the cNMF run (must match name used during inference) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| K | list of int | [30, 50, 70, 80, 100, 200, 300] | K values (number of components) to test |
| sel_threshs | list of float | [0.2, 2.0] | Density threshold values for consensus selection |
| categorical_key | str | "sample" | Key in .obs for cell condition/sample labels |

### Covariate Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| covariates | list of str | None | Covariate keys in .obs to include as-is (e.g., biological_sample) |
| log_covariates | list of str | None | Covariate keys in .obs to log1p-transform before inclusion (e.g., guide_umi_counts total_counts) |

### Calibration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| number_guide | int | 6 | Number of non-targeting guides to randomly designate as "targeting" in each calibration iteration |
| number_permutations | int | 1024 | Number of calibration iterations to run |
| guide_annotation_key | list of str | "non-targeting" | Name of target label for non-targeting/safe-targeting guides |
| FDR_method | str | "BH" | FDR correction method: "BH" (Benjamini-Hochberg) or "StoreyQ" (Storey Q-value) |
| save_dir | str | None | Directory to save results and figures. If not provided, defaults to `<out_dir>/<run_name>/Evaluation/<K>_<sel_thresh>/` |
| skip_existing | flag | off | If set, skip the CRT recompute for any (K, sel_thresh, condition) whose **both** result files (real `.txt` and fake `.txt`) already exist, and instead **regenerate the QQ `.png` from the cached raw p-values**. Use to resume a preempted job or to re-plot without recomputing. |

## Outputs

Per (K, sel_thresh, condition) CRT writes three files into the output folder:

| File | Contents |
|------|----------|
| `{K}_CRT_{covar_tag}_{condition}.txt` | **Real** perturbation results — columns: `target_name, program_name, log2FC, p-value` (skew-calibrated), `adj_pval` (FDR of skew), `p-value_raw` (raw CRT), `adj_pval_raw` (FDR of raw) |
| `{K}_CRT_fake_{covar_tag}_{condition}.txt` | **Fake / NTC null** distribution — columns: `ensemble, target_name` (NTC pseudo-gene id, e.g. `ntc_3`), `program_name, p-value_raw, adj_pval_raw` (raw p-values only, no effect size) |
| `{K}_CRT_{covar_tag}_{condition}.png` | QQ plot of real (raw) vs NTC-null (raw) p-values |

Real and null are saved with **raw** p-values so they are directly comparable (the QQ plot and any downstream calibration use the raw scale); the skew-calibrated p-value is retained on the real file for significance calls.

### Resuming / re-plotting a preempted job

To resume after preemption, append `--skip_existing` to the `python3 CRT.py ...` invocation in `CRT.sh` and resubmit — a (K, sel_thresh, condition) whose real **and** fake `.txt` both exist is not recomputed; its QQ `.png` is regenerated from the cached raw p-values (so plots stay fresh without paying for the CRT recompute). If either `.txt` is missing, that condition is recomputed. Without `--skip_existing`, every condition is recomputed and overwritten.