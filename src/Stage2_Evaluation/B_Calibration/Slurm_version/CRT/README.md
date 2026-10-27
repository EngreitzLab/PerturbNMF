# CRT

## Overview
A pipeline for testing differential effect (DE) of gene targets on gene programs. This pipeline implements a SCEPTRE-style Conditional Randomization Test (CRT) to assess the statistical significance of perturbations, and checks its calibration against a non-targeting control (NTC) null.

The CRT logic lives in the local `CRT` package at `src/Stage2_Evaluation/B_Calibration/src/CRT/`. It is a rewrite that replaces the earlier programDE-derived package (`run_all_genes_union_crt`, CLR outcome only, a fixed number of permutations, NTC ensembles; adapted from https://github.com/edtnguyen/programDE). `CRT.py` imports it directly by inserting `B_Calibration/src` at the front of `sys.path` — no external/PyPI install is required.

Public API (exported from `src/CRT/__init__.py`):

| Function | Purpose |
|----------|---------|
| `load_modality` | Read one modality of a `cNMF_<K>_<thresh>.h5mu`; drop cells with a missing covariate |
| `split_by_condition` | Yield one MuData per level of `--categorical_key` |
| `prepare_crt_inputs` | Build covariates, normalized outcome, guide assignment and precomputed OLS pieces (`CRT_obj`) |
| `conditional_randomized_test` | Run the CRT / permutation test for every selected gene (parallel over genes) |
| `prepare_null_guide_assignment` | Build NTC fake ("null") genes, returned as a new `CRT_obj` that runs through the same test |
| `format_results_table` / `load_results_table` | Long gene × program table in the Stage 3 format below (with FDR and log2FC) / reload it for `--skip_existing` |
| `plot_qq_real_vs_null` | QQ plots of real vs NTC-null p-values |
| `encode_categorical_covariates`, `get_covar_matrix`, `adjust_pvalues` | Covariate encoding, design matrix, BH / Storey Q adjustment |

`CRT.py` loops over every `--K` × `--sel_threshs`, and runs the CRT **separately per condition** (`--categorical_key`, via `split_by_condition`):

```python
mdata = load_modality(h5mu, "cNMF", covariate_cols)
for condition, mdata_con in split_by_condition(mdata, "cNMF", categorical_key):
    obj = prepare_crt_inputs(mdata_con, ...)
    conditional_randomized_test(obj, n_jobs=...)
    null_obj = prepare_null_guide_assignment(obj, ...)
    conditional_randomized_test(null_obj, n_jobs=...)
    format_results_table(obj); plot_qq_real_vs_null(obj, null_obj, ...)
```

The notebook `B_Calibration/JupterNote_Version/CRT.ipynb` mirrors `CRT.py` step by step for one condition (it imports the output writers from `CRT.py`). Tests: `tests/Script/Stage2_Evaluation/test_crt.py` (run in `NMF_Benchmarking`).

## Setup (required)

CRT is written in Python and runs in the **`NMF_Benchmarking`** conda env — the **same env as the Stage 2a evaluation metrics and the U-test calibration**. If you already have that env, no separate install is needed. Otherwise create it from the sibling `environment.yml`, which contains every dependency the `CRT` package needs (`numba`, `scipy`, `statsmodels`, `scikit-learn`, `joblib`, `mudata`, `multipy` for Storey Q, ...):

```bash
conda env create -f environment.yml
conda activate NMF_Benchmarking
```

> **Note:** Earlier versions used a dedicated `programDE` env and installed the wrapper from GitHub, importing `src.sceptre`. Neither is needed anymore. `CRT.sh` / `CRT_parallel.sh` export `PYTHONPATH=.../PerturbNMF/src` and set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, because parallelism comes from `--n_jobs` (workers over genes).

---

## Method overview

This pipeline tests **candidate regulator → program usage** effects using a per-program linear model, and assigns significance by resampling the treatment indicator (CRT for high MOI, permutation for low MOI), followed by BH or Storey q-value FDR control (`--FDR_method`).

### Model

Let $i$ index cells and $k$ index programs. $Y_{i,k}$ is the normalized usage of program $k$ in cell $i$:

- `--normalization row_normalize` (**default**): the per-cell usage share $Y_{ik} = U_{ik} / \sum_j U_{ij}$ (no log, no floor).
- `--normalization clr` (old behaviour): the centered log-ratio of the shares floored at `--eps` (default `1e-3`) and renormalized.

$X_i$ is a binary indicator of whether any guide targeting the candidate regulator is present in cell $i$ (union over the gene's guides), and $C_{ij}$ are the covariate values for cell $i$: `--covariates` as-is, `--log_covariates` after `log1p`; categorical columns (and numeric columns with ≤ `--numeric_as_category_threshold` unique values) are one-hot encoded, continuous columns are z-scored, and an intercept is added. The coefficient $\beta_k$ captures the perturbation effect on program $k$, while each $\gamma_j$ captures the effect of covariate $j$. The fitted $\beta_k$ is the covariate-adjusted effect size of that regulator on the program.

**Compact form:**

```math
Y_{ik} = \beta_k X_i + \sum_j \gamma_j C_{ij} + \varepsilon_i
```

**Expanded form:**

```math
\begin{aligned}
Y_{ik} = \beta_k X_i
&+ \gamma_0
+ \gamma_1\,(\%\,\mathrm{mito})_i
+ \gamma_2\,(\mathrm{replicate})_i
+ \gamma_3\,(\mathrm{total\ counts})_i \\
&+ \gamma_4\,(\mathrm{guide\ UMI})_i
+ \gamma_5\,(\mathrm{guide\ number})_i
+ \gamma_6\,(\mathrm{gene\ number})_i
+ \varepsilon_i
\end{aligned}
```

$\beta_k$ is obtained by OLS (the covariate cross-products $C^\top C$ and $C^\top Y$ are precomputed once, so each refit only needs the treated cells). With `row_normalize`, $\beta_k$ is the covariate-adjusted difference in mean usage share; with `clr`, it is a shift in CLR coordinate $k$.

The `log2FC` output column is derived from $\beta_k$:

- `clr`: $\log_2\mathrm{FC} = \beta_k \cdot \frac{K}{K-1} / \ln 2$ (the CLR mean absorbs $1/K$ of the shift).
- `row_normalize`: $\log_2\mathrm{FC} = \log_2\!\left(\frac{\bar Y^{\mathrm{ctrl}}_k + \beta_k}{\bar Y^{\mathrm{ctrl}}_k}\right)$, where $\bar Y^{\mathrm{ctrl}}_k$ is the mean share in the control cells (`control_mean`).

### Conditional Randomization Test (CRT) — `--moi high` (default)

P-values compare the observed effect size to a null distribution of resampled effect sizes. For each gene, we first estimate the propensity score — the probability that a cell carries a guide targeting the candidate regulator given its covariates — with a logistic regression on the same covariates $C_i$. By default the propensity model is **unpenalized** (exact MLE, as in sceptre); `--propensity_penalty l2` uses a ridge fit and `old_perturbnmf` reproduces the old PerturbNMF fit (comparison only — its null is anti-conservative):

**Propensity model:**

```math
p_i = P(X_i = 1 \mid C_i) = \frac{1}{1 + e^{-t_i}},
```

where $t_i = \sum_j \gamma_j C_{ij}$ is the linear predictor from the propensity model. We then generate $B$ resampled guide assignments by drawing from the Bernoulli distribution defined by each cell's propensity score:

**Resampling step:**

```math
x_i^{(b)} \sim \mathrm{Bernoulli}(p_i), \qquad b = 1, 2, 3, \dots, B.
```

For each resample $b$, we refit the regression model with the resampled guide indicator to obtain a null effect size $\beta_k^{(b)}$:

**Null regression:**

```math
Y_{ik} = \beta_k^{(b)} x_i^{(b)} + \sum_j \gamma_j^{(b)} C_{ij} + \varepsilon_i.
```

Controls are all other cells by default. `--control_group ntc_cells` restricts the controls to cells carrying a control guide (`--guide_annotation_key`); `--control_group matched` uses propensity-score-matched controls (`--matching_ratio`, `--matching_caliper`, `--matching_distance`), and the CRT then runs on the treated + matched cells only.

### Permutation test — `--moi low`

For low-MOI screens the regression uses only the treated cells and the control-guide cells, and the null shuffles the treated label **within covariate strata** (keeping each stratum's treated count fixed). Continuous covariates stay in the regression but cannot form strata, so with them the permutation null is approximate (a warning is printed).

### Two-stage resampling and p-values

Resampling runs in two stages (as in sceptre):

1. **Stage 1:** $B_1$ = `--number_permutations` rounds (default 4999) per gene.
2. **Stage 2:** if any program's stage-1 p is below `--extra_rounds_p_threshold` (default 0.05), $B_2$ = `--extra_resampled_rounds` more rounds (default 49999; `0` disables) are drawn and pooled with stage 1.

The empirical two-sided p-value for program $k$ is the fraction of resampled effect sizes whose magnitude equals or exceeds the observed one, with the $+1$ correction in numerator and denominator ($B = B_1$, or $B_1 + B_2$ after stage 2; null rounds that are empty or collinear are left out):

**Empirical p-value:**

```math
P_k = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\!\left(\left|\beta_k^{(b)}\right| \ge \left|\beta_k^{(\mathrm{obs})}\right|\right)}{B + 1}.
```

Its floor is $1/(B+1)$, so with stage 1 alone it cannot go below ~2e-4. With `--pvalue_method skew_normal` (**default**), each program flagged in stage 1 additionally gets a two-sided p from a **skew-normal fitted to all of its null rounds** (sceptre's resampling approximation), which resolves p-values below the empirical floor. Fits that fail sceptre's goodness-of-fit checks fall back to the empirical p. `--pvalue_method empirical` uses the empirical p everywhere. Three p-values are written per test:

| Column | Meaning |
|--------|---------|
| `p-value` | **Final** CRT p: `pval_skewnorm` where the skew-normal was fit and accepted, else `pval_empirical`. Use this (and `adj_pval`) for significance calls |
| `pval_empirical` | Empirical resampling p (formula above), for every test |
| `pval_skewnorm` | Skew-normal tail p; NaN where not fit (stage-1 p ≥ threshold, rejected fit, or `--pvalue_method empirical`) |

To control the false discovery rate across the full set of regulator–program tests in a condition, the final and empirical p-values are each adjusted with `--FDR_method` (`BH` by default, or `StoreyQ`), giving `adj_pval` and `adj_pval_empirical`. Regulator–program pairs with adjusted p < 0.05 are typically considered significant.

### Null calibration via NTC fake genes

To check that the CRT p-values are well calibrated, we build a **negative-control null** from the non-targeting control (NTC) guides and compare it against the real target p-values on a QQ plot (`--skip_null` turns this off).

1. **Fake genes** (`prepare_null_guide_assignment`). Each fake gene `null_<i>` is a set of `--number_guide` NTC guides. With `--null_method frequency` (**default**) the sets are frequency-matched to real genes (see below) over `--n_ensemble` random NTC partitions (default 10); with `--null_method random` they are sceptre-style random, distinct NTC guide sets, one per tested real gene.
2. **Null p-values.** The fake genes run through exactly the same test as the real genes (same normalization, covariates, MOI mode, two-stage resampling and p-value method).
3. **QQ diagnostic** (`plot_qq_real_vs_null`). Expected vs observed $-\log_{10}(p)$ for the real genes (red) against the NTC null (grey), all programs pooled, once for the empirical p and once for the final (skew-normal) p. Well-calibrated null points track the $y=x$ diagonal; real points rising above it indicate genuine perturbation signal. `--qq_per_program` adds a grid with one panel per program.

#### How NTC guides are binned against real guides to build fake genes

The NTC null is meaningful only if each fake gene looks like a *real* gene in the one respect that drives its CRT statistic: the per-guide prevalence of the guides it is built from (rarely-detected guides tag few cells and produce noisier, differently-distributed effect sizes than common guides). With `--null_method frequency` the pipeline does the following (`build_ntc_group_inputs` → `make_ntc_groups_matched_by_freq` in `src/CRT/ntc_groups.py`):

1. **Per-guide prevalence** (`guide_frequency`). For *every* guide $g$ — real and NTC alike — compute $\mathrm{freq}(g) = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{G_{ig} > 0\}$, the fraction of cells (in this condition) in which that guide is detected.

2. **Bin edges from the real guides only** (`_guide_bins_from_real_freqs`). Take the prevalences of the **real (targeting) guides** and compute `--n_bins` (default 10) quantile edges over that distribution. Every guide — real *and* NTC — is then assigned a bin index by digitizing its own prevalence against these shared edges. Because the edges come from the real-guide distribution, an NTC guide's bin tells you which real guides it is prevalence-comparable to.

3. **Real-gene "bin signature"** (`_real_gene_bin_signatures`). For each real gene with at least `--number_guide` guides, sort its guides by prevalence, keep the first `--number_guide` of them, and record the multiset of their bin indices. This vector of bins is the gene's *bin signature* — a compact fingerprint of "this gene is targeted by guides of these prevalence levels."

4. **Match NTC guides to a real gene's signature** (`make_ntc_groups_matched_by_freq`). Pool the NTC guides by bin. Repeatedly draw a real-gene bin signature at random and try to assemble a fake gene by pulling NTC guides from exactly the bins that signature calls for (e.g. a signature `[2,2,5,7,7,9]` pulls two NTC guides from bin 2, one from bin 5, two from bin 7, one from bin 9). Guides are drawn without replacement within a replicate (a partition of the NTC guides); if a bin is short of NTC guides the draw is skipped and another signature is tried. Each successful group therefore carries the **same size and the same per-guide prevalence composition as some real gene** — so its union of tagged cells, propensity model, and resulting CRT p-value are directly comparable to a real gene's.

5. **Ensemble over seeds.** Repeat the matched partitioning `--n_ensemble` times (seeds `seed`, `seed+1`, ...), drop repeated guide sets, and name the resulting fake genes `null_0`, `null_1`, ... so the null is averaged over many random NTC→real-gene assignments rather than one arbitrary partition.

The p-values of these fake genes are the **NTC null p-values** written to `{K}_CRT_fake_{covar_tag}_{condition}.txt` and plotted (grey) against the real target p-values on the QQ diagnostic.

> **Note:** `--number_guide` must match how many guides-per-gene your real targets use; real genes with fewer guides do not contribute signatures.

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
| save_dir | str | None | Base directory under which `{K}_{thresh}` subdirs are created. Default: `<out_dir>/<run_name>/Evaluation/` |
| skip_existing | flag | off | Reuse the per-(K, sel_thresh, condition) `{K}_CRT_*.txt` / `{K}_CRT_fake_*.txt` tables already in the output folder instead of recomputing; tables and QQ plots are rewritten (see below) |
| prog_key | str | "cNMF" | Modality with program usage (`.X`), `obsm["guide_assignment"]`, `uns["guide_names"]` / `uns["guide_targets"]` and covariates |
| categorical_key | str | "sample" | Key in .obs holding each cell's condition label (e.g. timepoint, stimulus, donor); the CRT runs separately per condition |
| n_jobs | int | `$SLURM_CPUS_PER_TASK` or 1 | Parallel workers over genes |

### Covariate Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| covariates | list of str | None | Covariate keys in .obs to include as-is (continuous or categorical) |
| log_covariates | list of str | None | Covariate keys in .obs to log1p-transform before inclusion (e.g. count-based values such as total_counts, guide UMI counts) |
| numeric_as_category_threshold | int | 20 | Numeric covariates with ≤ this many unique values are one-hot encoded (CRT design and `--check_covariate`) |
| check_covariate | flag | off | Before the CRT, run the covariate independence diagnostic (Pearson correlation heatmap + VIF) on the encoded design and write it to `<Evaluation>/covariate_check/` |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| moi | str | "high" | `high`: CRT (Bernoulli resampling from a per-gene propensity model, controls = all other cells); `low`: permutation test against control-guide cells within covariate strata |
| normalization | str | "row_normalize" | Outcome: per-cell usage share (`row_normalize`) or CLR of floored usage (`clr`). Also sets how log2FC is derived from beta |
| eps | float | 1e-3 | Share floor before CLR (`normalization=clr` only) |
| propensity_penalty | str | "none" | moi=high propensity model: `none` (unpenalized MLE, as sceptre), `l2` (ridge), or `old_perturbnmf` (old fit, comparison only) |
| control_group | str | "all" | Control cells per gene: `all` other cells, `ntc_cells` (cells with a control guide; moi=high), or `matched` (propensity-matched) |
| matching_ratio | int | 4 | `control_group=matched`: controls per treated cell |
| matching_caliper | float | None | `control_group=matched`: caliper in SDs of the distance (None = no caliper) |
| matching_distance | str | "propensity" | `control_group=matched`: match on the `propensity` score or its `logit` |

### Resampling / P-value Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| number_permutations | int | 4999 | Stage-1 resampling rounds per gene |
| extra_resampled_rounds | int | 49999 | Stage-2 rounds added for programs with stage-1 p < `--extra_rounds_p_threshold`; 0 disables |
| extra_rounds_p_threshold | float | 0.05 | Programs with stage-1 p below this get the stage-2 rounds and the skew-normal fit |
| pvalue_method | str | "skew_normal" | Final p-value: `empirical` resampling p, or sceptre-style `skew_normal` tail fit |
| FDR_method | str | "BH" | FDR correction method: "BH" (Benjamini-Hochberg) or "StoreyQ" (Storey Q-value) |
| seed | int | 123 | Base seed; each gene resamples with `crc32(gene) ^ seed` |

### Calibration (NTC null) Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| guide_annotation_key | list of str | "non-targeting" | Name(s) of the non-targeting/safe-targeting target in `uns["guide_targets"]` (target labels, not individual guide names) |
| number_guide | int | 6 | NTC guides per fake (null) gene |
| null_method | str | "frequency" | `frequency`: NTC groups matched to real genes' guide-frequency bins; `random`: sceptre-style random NTC guide sets |
| n_bins | int | 10 | `null_method=frequency`: guide-frequency bins |
| n_ensemble | int | 10 | `null_method=frequency`: number of random NTC partitions |
| skip_null | flag | off | Real genes only; no NTC null, fake file or QQ plots |
| qq_per_program | flag | off | Also save QQ plots with one panel per program |

## Outputs

`config_<jobid>.yml` (script args + SLURM info) is written to the save dir (`--save_dir`, or `<out_dir>/<run_name>/Evaluation/`), plus `covariate_check/` with `--check_covariate`. Per (K, sel_thresh, condition), CRT writes into `<save_dir>/{K}_{thresh}/` (`covar_tag` = `--covariates`, then `log_`-prefixed `--log_covariates`, joined by `_`; `no_covariates` when none):

| File | Contents |
|------|----------|
| `{K}_CRT_{covar_tag}_{condition}.txt` | **Real** perturbation results (TSV, read by Stage 3) — columns: `target_name, program_name` (1-based integer), `log2FC, p-value` (final: `pval_skewnorm` where fit, else `pval_empirical`), `adj_pval` (FDR of final), `pval_empirical, adj_pval_empirical, pval_skewnorm` (NaN where not fit), `beta, n_treated, n_control, control_mean, n_resampled`, `propensity_n_iter, propensity_sum_ratio` (high-MOI propensity fit: lbfgs iterations, = max_iter means it hit the cap; sum(p)/n_treated, 1 when converged; NaN for low MOI) |
| `{K}_CRT_fake_{covar_tag}_{condition}.txt` | **Fake / NTC null** — columns: `target_name` (fake gene id, e.g. `null_3`), same columns as the real table, plus `guides` (`;`-joined NTC guides). The old `ensemble` column is gone |
| `{K}_CRT_{covar_tag}_{condition}_qq_empirical.pdf` | QQ plot of real (red) vs NTC-null (grey) empirical p-values |
| `{K}_CRT_{covar_tag}_{condition}_qq_skew_normal.pdf` | Same for the final p (skew-normal where fit); only with `--pvalue_method skew_normal` |
| `..._qq_{empirical,skew_normal}_per_program.pdf` | One panel per program (with `--qq_per_program`) |

### Resuming / re-plotting a preempted job

To resume after preemption, append `--skip_existing` to the `python3 CRT.py ...` invocation in `CRT.sh` and resubmit. For each (K, sel_thresh, condition), the existing `{K}_CRT_*.txt` (and `{K}_CRT_fake_*.txt`) table is loaded (`load_results_table`) instead of rerunning the CRT, and the tables (so `adj_pval` follows `--FDR_method`) and QQ plots are rewritten from it. The cached tables must have been made with the same settings: genes and programs must match, and the null guide sets are checked against the newly built ones (a null cached with another `--number_guide` / `--null_method` / `--seed` is rejected). If a table is missing, that part is recomputed. Without `--skip_existing`, every condition is recomputed and overwritten.
