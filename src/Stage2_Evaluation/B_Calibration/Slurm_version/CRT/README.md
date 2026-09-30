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
\begin{aligned}
Y_{ik} = \beta_k X_i
&+ \gamma_1\,(\%\,\mathrm{mito})_i
+ \gamma_2\,(\mathrm{replicate})_i
+ \gamma_3\,(\mathrm{total\ counts})_i \\
&+ \gamma_4\,(\mathrm{guide\ UMI})_i
+ \gamma_5\,(\mathrm{guide\ number})_i
+ \gamma_6\,(\mathrm{gene\ number})_i
+ \varepsilon_i
\end{aligned}
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

#### Fixed-count resampling (`--resampling fixed_count`)

With Bernoulli resampling the size of each null treated set is random, with mean $\sum_i p_i$ rather than the observed count $n$. For few-cell targets the unstandardized $\beta$ is dominated by single extreme CLR values (a program with zero usage, floored at $10^{-8}$, gives CLR $\approx -18$), so its null distribution depends strongly on set size: larger null sets dilute extreme cells and empty ones give $\beta = 0$. In a CRISPRi Perturb-seq screen (K=60, lane covariate), a cell-count-matched negative-control null showed Bernoulli CRT p-values are anticonservative (null $\lambda = 1.19$; 31% of null tests at $p < 0.05$ for targets with $\le 10$ cells).

`--resampling fixed_count` draws every null treated set with exactly the observed treated count in each **covariate stratum** (cells with identical values of all discrete design columns, i.e. the one-hot encoded categorical covariates such as lane):

- **Categorical covariates only** — target labels are shuffled uniformly within each stratum. The design rows of the treated set are then identical in every permutation, so this is an exact permutation test conditional on the per-stratum counts, and no propensity model is fit. (Matched null: $\lambda = 1.00$, 3.4% at $p < 0.05$ for $\le 10$ cells.)
- **With continuous covariates** — within each stratum, exactly $n_s$ cells are drawn by Pareto sampling (keys $\frac{U_j}{1-U_j}\cdot\frac{1-p_j}{p_j}$, $U_j \sim \mathrm{Unif}(0,1)$, keep the $n_s$ smallest), a standard fixed-size approximation to conditional Poisson sampling with weights $p_j/(1-p_j)$. This is **approximate** and costs $O(N)$ per resample.

The observed $\beta$ is identical under both options; only the null distribution changes.

#### Outcome: CLR or usage share (`--outcome`)

By default (`--outcome clr`) $Y_{ik}$ is the centered log-ratio of cell $i$'s usage, floored at $10^{-8}$ and renormalized, and the effect is reported as `log2FC` (see above). With `--outcome usage`, $Y_{ik} = U_{ik} / \sum_{k'} U_{ik'}$ is the per-cell **usage share** of program $k$: usage is row-normalized to proportions (no log, no floor, zeros stay 0). Row-normalizing matters because some staged h5mu files hold raw, depth-scaled cNMF usages whose row sums track library size. The design, covariates, resampling and p-values are unchanged; $\beta_k$ becomes the covariate-adjusted difference in mean usage share (treated minus control, on the 0–1 scale) and is written as `usage_share_diff`, not `log2FC`. Two more columns give a relative effect: `control_usage_share`, the covariate-adjusted mean share the treated cells would have without the perturbation ($\bar{C}_T \hat\gamma$), and `usage_share_relative_diff` $= \beta / $ `control_usage_share` (a fold change minus 1 on mean usage; NaN if the baseline is $\le 0$).

What changes with the usage scale:
- **Additive, not multiplicative.** 0.02→0.03 and 0.40→0.60 are both 1.5× on the CLR scale, but +0.01 vs +0.20 in `usage_share_diff`. Sorting by `usage_share_diff` favors abundant programs; use `usage_share_relative_diff` to compare programs with different baselines. p-values are per program, so validity is unaffected.
- **Arithmetic vs geometric mean.** The usage mean is driven by high-usage cells; CLR is driven by the typical cell, including near-zero ones. Usage is more sensitive to gains in abundant programs, less to a program switching off in cells where it was already low.
- **Zeros.** 0 stays 0 instead of becoming a floor-dependent outlier (CLR $\approx -18$ at a $10^{-8}$ floor), which is why no floor hyperparameter is needed.
- **Compositional in both cases.** Shares sum to 1, so a gain in one program forces losses elsewhere (CLR has the same constraint).
- **Downstream thresholds** written for `log2FC` do not apply to `usage_share_diff`.

In an internal benchmark (K=60, lane covariate, 5,000 permutations), judged against a cell-count-matched, guide-structured NTC null, `--resampling fixed_count --outcome usage` was calibrated (null $\lambda = 1.01$) and found as many regulators as the best CLR floor, without a floor hyperparameter; the $10^{-8}$ floor used by `--outcome clr` found ~40% fewer. **Recommended: `--resampling fixed_count --outcome usage`.**

The empirical two-sided p-value for program $k$ was then computed as the fraction of resampled effect sizes whose magnitude equaled or exceeded the observed effect size, with the standard $+1$ correction in both numerator and denominator to prevent p-values of zero:

**Empirical p-value:**

```math
P_k = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\!\left(\left|\beta_k^{(b)}\right| \ge \left|\beta_k^{(\mathrm{obs})}\right|\right)}{B + 1}.
```

To control the false discovery rate (FDR) across the full set of regulator–program tests, we applied the Storey q-value procedure. Regulator–program pairs with $q < 0.05$ were considered to be significant.

### Null calibration via NTC guide-group ensembles

To check that the CRT p-values are well calibrated, we generate a **negative-control null** from the non-targeting control (NTC) guides and compare it against the real target p-values on a QQ plot.

The null is built to look like real targets rather than being drawn arbitrarily:

1. **Frequency binning** (`build_ntc_group_inputs`). Compute each guide's prevalence $\mathrm{freq}(g)=\text{mean}(G[:,g]>0)$. Bin the *real* (targeting) guides into `n_bins` frequency quantiles, and record, for every real gene, the frequency-bin composition ("bin signature") of its guide set.

2. **Matched NTC groups** (`make_ntc_groups_ensemble` → `make_ntc_groups_matched_by_freq`). Randomly partition the NTC guides into synthetic "pseudo-gene" groups of size `--number_guide`, choosing guides so each group's frequency-bin composition matches a randomly drawn real-gene bin signature. This is repeated `n_ensemble` times (different seeds) to build an ensemble of null groupings. Because the NTC groups match real genes in both group size and per-guide prevalence, they form a fair null.

3. **Null p-values** (`crt_pvals_for_ntc_groups_ensemble_skew_and_raw`). Run the exact same union-CRT (same `--resampling`, raw permutation p **and** the skew-normal p that real-target calls use) on each NTC pseudo-gene group across all programs, producing a null distribution of p-values.

4. **QQ diagnostic** (`qq_plot_real_vs_null`). Plot expected vs observed $-\log_{10}(p)$ for the real target p-values (purple) against the NTC null (blue). Well-calibrated null points track the $y=x$ diagonal; real points rising above it indicate genuine perturbation signal. One `{K}_CRT_{covariates}_{condition}.png` is written per (K, sel_thresh, condition).

#### How NTC guides are binned against real guides to build pseudo-genes

The NTC null is meaningful only if each synthetic NTC "pseudo-gene" looks like a *real* gene in the one respect that drives its CRT statistic: the per-guide prevalence of the guides it is built from (rarely-detected guides tag few cells and produce noisier, differently-distributed effect sizes than common guides). To point each NTC pseudo-gene at a real gene's prevalence profile, the pipeline does the following (all in `build_ntc_group_inputs` → `make_ntc_groups_matched_by_freq`):

1. **Per-guide prevalence** (`guide_frequency`). For *every* guide $g$ — real and NTC alike — compute $\mathrm{freq}(g) = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{G_{ig} > 0\}$, the fraction of cells in which that guide is detected.

2. **Bin edges from the real guides only** (`_guide_bins_from_real_freqs`). Take the prevalences of the **real (targeting) guides** and compute `n_bins` (default 20) quantile edges over that distribution. Every guide — real *and* NTC — is then assigned a bin index by digitizing its own prevalence against these shared edges. Because the edges come from the real-guide distribution, an NTC guide's bin tells you which real guides it is prevalence-comparable to.

3. **Real-gene "bin signature"** (`_real_gene_bin_signatures`). For each real gene, take its guides, sort them by prevalence, keep the first `--number_guide` of them, and record the multiset of their bin indices. This length-`--number_guide` vector of bins is the gene's *bin signature* — a compact fingerprint of "this gene is targeted by guides of these prevalence levels."

4. **Match NTC guides to a real gene's signature** (`make_ntc_groups_matched_by_freq`). Pool the NTC guides by bin. Repeatedly draw a real-gene bin signature at random and try to assemble a pseudo-gene by pulling NTC guides from exactly the bins that signature calls for (e.g. a signature `[2,2,5,7,7,9]` pulls two NTC guides from bin 2, one from bin 5, two from bin 7, one from bin 9). Guides are drawn without replacement within a replicate (default), or with `--ntc_replace` returned to the pool after each group so a guide can appear in several groups but never twice in one, with identical groups rejected. The latter follows SCEPTRE's negative-control group construction (`sample_combinations_v2`, [Katsevich-Lab/sceptre](https://github.com/Katsevich-Lab/sceptre)) and keeps the null from running out of groups when guides per target are many relative to NTC guides. If a bin is short of NTC guides the draw is skipped and another signature is tried. Each successful group is labelled `ntc_{idx}` and, by construction, carries the **same size and the same per-guide prevalence composition as some real gene** — so its union of tagged cells, propensity model, and resulting CRT p-value are directly comparable to a real gene's.

5. **Ensemble over seeds** (`make_ntc_groups_ensemble`). Repeat the matched partitioning `n_ensemble` times with different seeds so the null is averaged over many random NTC→real-gene assignments rather than one arbitrary partition.

The p-values from step 5 are exactly the **NTC null p-values** written to `{K}_CRT_fake_{covar_tag}_{condition}.txt` and plotted (blue) against the real target p-values on the QQ diagnostic.

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
| resampling | str | "bernoulli" | How null treated sets are drawn: "bernoulli" (each cell ~ Bernoulli(propensity); set size varies) or "fixed_count" (treated count fixed within each categorical-covariate stratum; see *Fixed-count resampling*). Output file names do not change with this flag — use `--save_dir` to keep runs apart |
| outcome | str | "clr" | CRT outcome $Y$: "clr" (CLR of usage floored at 1e-8; effect column `log2FC`) or "usage" (per-cell usage share, row-normalized, no log/floor; effect column `usage_share_diff`, a difference in mean usage share). See *Outcome*. Output file names do not change — use `--save_dir` to keep runs apart |
| matched_ntc_null | flag | off | Also build a cell-count-matched NTC null: one NTC pseudo-target per real target with exactly its cell count (whole NTC guides in random order, topped up from the next guide), scored with the same CRT. Writes `{K}_CRT_matched_null_{covar_tag}_{condition}.txt` and prints the share of null p < 0.05 by cells per target. Roughly doubles runtime. Not recomputed under `--skip_existing` |
| ntc_replace | flag | off | Build NTC null groups with replacement across groups: a guide can appear in several groups (never twice in one), and identical groups are rejected. This is how SCEPTRE builds its negative-control gRNA groups. Without it, guides are consumed, so each ensemble gets at most ~n_ntc / `--number_guide` groups (fewer after frequency-bin matching), which is too few when `--number_guide` is large. With no `--max_ntc_groups`, builds one group per real target per ensemble (~10x the real-target CRT runtime) |
| max_ntc_groups | int | None | Cap on NTC null groups per ensemble. None = as many as the pool allows (with `--ntc_replace`: one per real target) |
| save_dir | str | None | Directory to save results and figures. If not provided, defaults to `<out_dir>/<run_name>/Evaluation/<K>_<sel_thresh>/` |
| skip_existing | flag | off | If set, skip the CRT recompute for any (K, sel_thresh, condition) whose **both** result files (real `.txt` and fake `.txt`) already exist, and instead **regenerate the QQ `.png` from the cached raw p-values**. Use to resume a preempted job or to re-plot without recomputing. |

## Outputs

Per (K, sel_thresh, condition) CRT writes three files into the output folder:

| File | Contents |
|------|----------|
| `{K}_CRT_{covar_tag}_{condition}.txt` | **Real** perturbation results — columns: `target_name, program_name, log2FC` (with `--outcome usage`: `usage_share_diff, control_usage_share, usage_share_relative_diff`), `p-value` (skew-calibrated), `adj_pval` (FDR of skew), `p-value_raw` (raw CRT), `adj_pval_raw` (FDR of raw) |
| `{K}_CRT_fake_{covar_tag}_{condition}.txt` | **Fake / NTC null** distribution — columns: `ensemble, target_name` (NTC pseudo-gene id, e.g. `ntc_3`), `program_name, p-value` (skew-calibrated, the p the real calls use), `adj_pval`, `p-value_raw, adj_pval_raw` (no effect size) |
| `{K}_CRT_{covar_tag}_{condition}.png` | QQ plot of real (raw) vs NTC-null (raw) p-values |
| `{K}_CRT_{covar_tag}_{condition}_skew.png` | QQ plot of real vs NTC-null **skew-calibrated** p-values (the scale significance calls are made on) |
| `{K}_CRT_ntc_significance_{covar_tag}_{condition}[_skew].txt` | NTC empirical-FDR check on `adj_pval_raw` (and, `_skew`, on `adj_pval`) |
| `{K}_CRT_matched_null_{covar_tag}_{condition}.txt` | Only with `--matched_ntc_null` — `matched_target, n_cells, n_guides, program_name, p-value, p-value_raw` |

Real and null carry both **raw** and **skew-calibrated** p-values. Significance calls (`adj_pval`) use the skew-calibrated p, so judge calibration on the `_skew` QQ plot / NTC summary; the raw-scale outputs are unchanged from earlier versions. Two-sided skew-normal p-values are clipped at 1.

### Resuming / re-plotting a preempted job

To resume after preemption, append `--skip_existing` to the `python3 CRT.py ...` invocation in `CRT.sh` and resubmit — a (K, sel_thresh, condition) whose real **and** fake `.txt` both exist is not recomputed; its QQ `.png` is regenerated from the cached raw p-values (so plots stay fresh without paying for the CRT recompute). If either `.txt` is missing, that condition is recomputed. Without `--skip_existing`, every condition is recomputed and overwritten.