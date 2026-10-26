# Calibration Stage (CRT / U-test / Matched Cell DE)

> **All-flags convention (mandatory for every generated `.sh`):** When invoking `generate_slurm.py`, list active flags first, then `---COMMENTED---`, then every remaining flag for this stage from `references/parameter-catalog.md` (Sections 7–9 — U-test, CRT, Matched Cell DE) with a sensible default/example value. The generator emits unused flags as `#     --flag value` lines below the python command so the user can toggle them later. See `SKILL.md` Step 5.

---

## CRT Calibration

### Step A: Determine guide data format

Check the h5mu structure to identify how guide/perturbation data is stored:

```bash
cat <out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>_structure.txt
```

CRT reads guide info (`obsm['guide_assignment']`, `uns['guide_names']`, `uns['guide_targets']`) directly from the cNMF modality of each `cNMF_<K>_<thresh>.h5mu` — no separate `mdata_guide_path` flag (formerly required) and no preprocessing needed.

If the h5mu uses alternative keys (e.g. `obsm['target_assignment']`, `uns['target_names']`), rename them into the cNMF modality before running CRT.

### Step B: Determine experiment type

| Experiment Type | `--guide_annotation_key` | `--number_guide` | Notes |
|-----------------|--------------------------|-------------------|-------|
| CRISPR screen (multi-guide per gene) | `non-targeting` | Number of guides per gene (e.g., `6`) | Standard Perturb-seq |
| KO village (gene knockouts) | `WT` | `1` | Each target is a single gene KO |
| CRISPRi with safe-targeting | `safe-targeting` | Guides per gene | Adjust label to match data |

Ask the user about their experiment type to determine these values.

Also ask the **MOI** (sets `--moi`):

| Screen | `--moi` | Test |
|--------|---------|------|
| High MOI (many cells carry several guides) | `high` (default) | CRT: per-gene propensity model, treated sets resampled Bernoulli(p), controls = all other cells (`--control_group` can narrow this) |
| Low MOI (≲1 guide per cell; enough control-guide cells) | `low` | Permutation test of each gene's cells against the control-guide cells, within covariate strata |

If unsure, count guides per cell from `obsm['guide_assignment']` (or the `guides_per_cell` obs column).

### Step C: CRT parameters

Required:
- `--out_dir`: Parent directory containing the run
- `--run_name`: cNMF run name

Key parameters (always ask):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--K` | `[30, 50, 70, 80, 100, 200, 300]` | K values to test (formerly named `--components`) |
| `--sel_threshs` | `[0.2, 2.0]` | Density thresholds |
| `--categorical_key` | `sample` | Key in .obs holding each cell's condition label (e.g. timepoint, stimulus, donor); CRT runs per condition |
| `--guide_annotation_key` | `non-targeting` | Label for non-targeting/control guides (see Step B) |
| `--moi` | `high` | `high`: CRT (Bernoulli resampling from a per-gene unpenalized propensity model, controls = all other cells); `low`: permutation test vs control-guide cells within covariate strata |
| `--normalization` | `row_normalize` | Outcome: per-cell usage share, or `clr` (old behaviour) |
| `--number_permutations` | `4999` | Stage-1 resampling rounds per gene |
| `--extra_resampled_rounds` | `49999` | Stage-2 rounds for programs with stage-1 p < `--extra_rounds_p_threshold` (`0.05`); `0` disables |
| `--pvalue_method` | `skew_normal` | Final p: skew-normal tail fit for flagged programs (empirical elsewhere), or `empirical` |
| `--number_guide` | `6` | NTC guides per fake (null) gene (see Step B) |
| `--covariates` | None | obs columns to include as-is (e.g. `pct_counts_mt doublet_scores`) |
| `--log_covariates` | None | obs columns to log1p-transform before inclusion (e.g. `total_counts n_genes`) |
| `--FDR_method` | `BH` | FDR correction: `BH` or `StoreyQ` |
| `--n_jobs` | `$SLURM_CPUS_PER_TASK` or 1 | Parallel workers over genes |
| `--save_dir` | auto | Base directory for the `<K>_<sel_thresh>/` folders (default: `<out_dir>/<run_name>/Evaluation/`) |
| `--skip_existing` | off | Reuse the `{K}_CRT_*.txt` / `{K}_CRT_fake_*.txt` tables already in the output folder instead of recomputing; tables and QQ plots are rewritten. Cached tables must come from the same settings (null guide sets are checked). Resume a preempted job. |

Advanced (defaults are usually right; see `parameter-catalog.md` Section 8): `--prog_key`, `--numeric_as_category_threshold`, `--check_covariate`, `--eps`, `--propensity_penalty` (`none`/`l2`/`old_perturbnmf`), `--control_group` (`all`/`ntc_cells`/`matched`) with `--matching_ratio` / `--matching_caliper` / `--matching_distance`, `--extra_rounds_p_threshold`, `--seed`, `--null_method` (`frequency`/`random`), `--n_bins` (10), `--n_ensemble` (10), `--skip_null`, `--qq_per_program`.

> **`--number_guide` controls the NTC null.** CRT calibrates its p-values by building non-targeting-control (NTC) fake genes of `--number_guide` guides, frequency-matched to real genes (`--null_method frequency`), running them through the same CRT, and comparing them to real targets on a QQ plot. Set `--number_guide` to your real guides-per-gene (a wrong value skews the null).

**Output files** in `<save_dir>/<K>_<sel_thresh>/` (covariate token *before* the condition, so different covariate combinations can share a `--save_dir`; no covariates → token is `no_covariates`):
- `{K}_CRT_<covariates>_{condition}.txt` — **real** results (Stage-3 format): `target_name, program_name` (1-based int), `log2FC, p-value` (**final**: `pval_skewnorm` where the skew-normal was fit, else `pval_empirical`), `adj_pval` (FDR of final), `pval_empirical` (resampling p), `adj_pval_empirical, pval_skewnorm` (NaN where not fit), `beta, n_treated, n_control, control_mean, n_resampled`.
- `{K}_CRT_fake_<covariates>_{condition}.txt` — **fake / NTC null**: `target_name` (`null_<i>`), same columns as the real table, plus `guides` (`;`-joined NTC guides). (Old `ensemble`, `p-value_raw`, `adj_pval_raw` columns are gone.)
- `{K}_CRT_<covariates>_{condition}_qq_empirical.pdf` / `_qq_skew_normal.pdf` — square QQ plots of real (red) vs NTC null (gray) -log10 p against Uniform, with y = x (dashed), a gray dotted line at p = 0.05 and a blue dash-dot line at the empirical p floor 1/(1 + all rounds); the title gives the null fraction p < 0.05. `--qq_per_program` adds per-program grids.

Covariate token = `--covariates` values, then each `--log_covariates` value prefixed with `log_`, joined by `_`.

Set `--save_dir` to the `Evaluation/` base directory (`<out_dir>/<run_name>/Evaluation/`) — CRT.py appends `<K>_<sel_thresh>` itself, and the covariate filename token keeps runs with different covariates separate within the same directory. Do **not** nest it under a `Calibration/CRT_<covariate_names>/` subdir.

### Step D: Select covariates

Present available continuous covariates from the h5mu obs columns. Common covariates by category:

| Category | Typical column names | Transform | Use as |
|----------|---------------------|-----------|--------|
| Library size | `total_counts`, `nCount_RNA`, `n_counts` | `--log_covariates` | log1p-transform (count data) |
| Gene count | `nFeature_RNA`, `n_genes`, `n_genes2` | `--log_covariates` | log1p-transform (count data) |
| MT percentage | `pct_counts_mt` | `--covariates` | Raw (already a percentage) |
| MT counts | `total_counts_mt` | `--log_covariates` | log1p-transform (count data) |
| Doublet score | `doublet_scores` | `--covariates` | Raw (already a probability) |
| Guides per cell | `guides_per_cell` | `--log_covariates` | log1p-transform (count data) |

**Redundancy warning**: Some datasets have multiple equivalent columns (e.g., `nCount_RNA` ~ `n_counts`, `nFeature_RNA` ~ `n_genes` ~ `n_genes2`). Use only one from each group to avoid multicollinearity.

### Step E: Resource estimation

| Factor | Recommendation |
|--------|---------------|
| CPUs | `40` (CRT parallelizes over genes with `--n_jobs`, default `$SLURM_CPUS_PER_TASK`) |
| Memory | `256G` for >100k cells, `128G` for <100k |
| Time | ~2h per K x sel_thresh x categorical condition. E.g., 1 K x 1 sel_thresh x 17 timepoints = 4-8h |
| Partition | a partition that allows high-memory jobs (ask the user) |

### Step F: Generate SLURM script

```bash
python3 SKILL_DIR/scripts/generate_slurm.py \
  --stage crt-calibration \
  --job_name CRT_<run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus 40 --mem 256G --time 04:00:00 \
  --partition <PARTITION> \
  --script_output_path <project_root>/Script/<run_name>_crt.sh \
  -- \
  --out_dir <out_dir> \
  --run_name <run_name> \
  --guide_annotation_key <ntc_label> \
  --number_permutations 4999 \
  --extra_resampled_rounds 49999 \
  --number_guide <N> \
  --K <K_values> \
  --sel_threshs <thresholds> \
  --categorical_key <key> \
  --log_covariates <cols...> \
  --covariates <cols...> \
  --moi high \
  --normalization row_normalize \
  --pvalue_method skew_normal \
  --FDR_method StoreyQ \
  --n_jobs 40 \
  --save_dir <out_dir>/<run_name>/Evaluation
```

> `--save_dir` must be the `Evaluation/` base directory. CRT.py appends `<K>_<sel_thresh>` on its own, so passing e.g. `Evaluation/Calibration/CRT_sample` would produce `Evaluation/Calibration/CRT_sample/<K>_<thresh>/` — wrong. Use `<out_dir>/<run_name>/Evaluation`.

Generated CRT scripts export `OMP/MKL/OPENBLAS_NUM_THREADS=1` (parallelism comes from `--n_jobs`); keep `--n_jobs` equal to `--cpus`.

If any requested covariate isn't present in `cNMF.obs`, add a small prep step before CRT that injects it (e.g. compute `guides_per_cell` from `obsm['guide_assignment']`). Mutate each `cNMF_<K>_<thresh>.h5mu` directly — CRT reads guide info from there.


### Step G: Check calibration and hand off to Stage 3

After the job finishes, check the QQ PDFs per condition: the gray NTC null should follow y = x (null fraction p < 0.05 should be ≲ 0.05; well above it means anti-conservative — revisit covariates, `--moi`, or `--control_group`), and real genes should rise above it. Empirical p cannot go below the blue floor line; hits stacked on it need more rounds (`--extra_resampled_rounds`) or `--pvalue_method skew_normal`.

Stage 3 plotting reads `{perturb_path_base}_{condition}.txt`, so for CRT pass
`--perturb_path_base <save_dir>/<K>_<thresh>/<K>_CRT_<covariate_token>` (e.g. `.../Evaluation/50_0_2/50_CRT_pct_counts_mt_log_total_counts`) with the default column names (`target_name`, `program_name`, `log2FC`, `adj_pval`). To resume a preempted job, resubmit the same script with `--skip_existing`.
---

## U-test Calibration

### Required parameters

- `--out_dir`, `--run_name`

U-test reads guide info (`obsm['guide_assignment']`, `uns['guide_names']`, `uns['guide_targets']`) directly from each `cNMF_<K>_<thresh>.h5mu` — no separate `mdata_guide_path` flag required (same pattern as CRT).

### Key optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--K` | `[30, 50, 70, 80, 100, 200, 300]` | K values (formerly named `--components`) |
| `--sel_threshs` | `[0.2, 2.0]` | Density thresholds |
| `--guide_annotation_key` | `['non-targeting']` | Non-targeting guide label |
| `--guide_annotation_path` | None | TSV with a `targeting` column (alternative to `--guide_annotation_key`) |
| `--reference_gtf_path` | None | Reference GTF for validating gene names during `--check_format` |
| `--number_run` | `300` | Calibration iterations |
| `--number_guide` | `6` | Fake targeting guides per iteration |
| `--FDR_method` | `StoreyQ` | FDR correction method |
| `--compute_real_perturbation_tests` | flag | Compute perturbation association on real targeting guides |
| `--compute_fake_perturbation_tests` | flag | Compute perturbation association on fake targeting guides (null) |
| `--visualizations` | flag | Generate QQ plots and violin plots comparing real vs null distributions |
| `--reference_mode` | str (default `complement`) | Reference cells for the real and fake tests: `complement` = all other cells, `ntc_only` = only control guides; each target's own cells are always removed from its reference |
| `--qq_per_program` | flag | With `--visualizations`, also save a QQ plot with one panel per program |
| `--balanced` | flag | Subsample each target's reference (real and fake tests) to the target's cell count; all reference cells kept if fewer |
| `--balanced_seed` | int (default `0`) | Seed for the `--balanced` subsample |
| `--check_format` | flag | Validate MuData format + required keys before running calibration |
| `--skip_existing` | flag | Resume preempted jobs by skipping per-(K, sel_thresh, sample) outputs that already exist on disk; visualizations still work via on-disk loads |

### SLURM resources

- Partition: standard CPU partition (ask the user)
- CPUs: 10-20, Memory: 128-256G, Time: 3-5h

---

## Matched Cell DE

Runs via R script. Conda env: `programDE`.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--input` | Input directory |
| `--output_dir` | Output directory for results |
| `--cell_metadata` | Path to cells_metadata.tsv |
| `--cnmf_usages` | cNMF usages file (program loadings) |
| `--gene_perturbations_sparse` | Guide assignment file |
| `--gene_batch_file` | File with batch gene names to test (one per line) |
| `--perturbation_names` | File with all gene names to test (one per line) |
| `--condition` | Condition to test |
| `--batch_id` | Batch identifier for output filename |
| `--script_dir` | Directory containing de_testing_utils.R |

### Key optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test_type` | `genes` | What to test: `genes` or `nulls` |
| `--min_cells_per_gene` | `50` | Minimum cells required per gene |
| `--matching_ratio` | `5` | Controls per treated cell |
| `--methods` | `ols_log` | DE methods (space-separated: `ols_log`, `ols_pseudo`, `hurdle_marginal`) |
| `--de_level` | `programs` | DE level: `programs`, `genes`, or `both` |

See `references/parameter-catalog.md` for the full parameter list including gene propagation and direct gene-level DE options.

### SLURM resources

- Partition: standard CPU partition (ask the user)
- CPUs: 10-20, Memory: 128-256G, Time: 2-6h
