# Calibration Stage (CRT / U-test / Matched Cell DE)

---

## CRT Calibration

### Step A: Determine guide data format

Check the h5mu structure to identify how guide/perturbation data is stored:

```bash
cat <out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>_structure.txt
```

CRT reads guide info (`obsm['guide_assignment']`, `uns['guide_names']`, `uns['guide_targets']`) directly from the cNMF modality of each `cNMF_<K>_<thresh>.h5mu` — no separate `--mdata_guide_path` and no preprocessing required.

If the h5mu uses alternative keys (e.g. `obsm['target_assignment']`, `uns['target_names']`), rename them into the cNMF modality before running CRT.

### Step B: Determine experiment type

| Experiment Type | `--guide_annotation_key` | `--number_guide` | Notes |
|-----------------|--------------------------|-------------------|-------|
| CRISPR screen (multi-guide per gene) | `non-targeting` | Number of guides per gene (e.g., `6`) | Standard Perturb-seq |
| KO village (gene knockouts) | `WT` | `1` | Each target is a single gene KO |
| CRISPRi with safe-targeting | `safe-targeting` | Guides per gene | Adjust label to match data |

Ask the user about their experiment type to determine these values.

### Step C: CRT parameters

Required:
- `--out_dir`: Parent directory containing the run
- `--run_name`: cNMF run name

Key parameters (always ask):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--components` | `[30,50,60,80,100,200,250,300]` | K values to test |
| `--sel_thresh` | `[0.4,0.8,2.0]` | Density thresholds |
| `--categorical_key` | `sample` | Key to split cells into conditions for per-condition CRT |
| `--guide_annotation_key` | `non-targeting` | Label for non-targeting/control guides (see Step B) |
| `--number_permutations` | `1024` | CRT permutations (recommend `5000` for production) |
| `--number_guide` | `6` | Guides per gene (see Step B) |
| `--FDR_method` | `BH` | FDR correction: `BH` or `StoreyQ` |
| `--save_dir` | auto | Custom output directory |

**Output filenames** include the covariate set so different combinations can share a `--save_dir`:
- `{K}_CRT_{condition}_<covariates>.txt`  (e.g. `30_CRT_1_percent_mito_log_n_counts.txt`)
- `{K}_CRT_{condition}_<covariates>.png` (QQ plot)
- No covariates → suffix is `_no_covariates`.

The recommended directory convention is `Evaluation/Calibration/CRT_<covariate_names>/` (one folder per covariate combination), but the filename suffix means runs with different covariates can also coexist in the same directory.

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
| CPUs | `40` (CRT parallelizes permutations across cores) |
| Memory | `256G` for >100k cells, `128G` for <100k |
| Time | ~2h per K x sel_thresh x categorical condition. E.g., 1 K x 1 sel_thresh x 17 timepoints = 4-8h |
| Partition | `owners,engreitz,bigmem` |

### Step F: Generate SLURM script

```bash
python3 SKILL_DIR/scripts/generate_slurm.py \
  --stage crt-calibration \
  --job_name CRT_<run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus 40 --mem 256G --time 04:00:00 \
  --partition owners,engreitz,bigmem \
  --log_dir <save_dir>/logs \
  --script_output_path <project_root>/Script/<run_name>_crt.sh \
  -- \
  --out_dir <out_dir> \
  --run_name <run_name> \
  --guide_annotation_key <ntc_label> \
  --number_permutations 5000 \
  --number_guide <N> \
  --components <K_values> \
  --sel_thresh <thresholds> \
  --categorical_key <key> \
  --log_covariates <cols...> \
  --covariates <cols...> \
  --FDR_method StoreyQ \
  --save_dir <save_dir>
```

If any requested covariate isn't present in `cNMF.obs`, add a small prep step before CRT that injects it (e.g. compute `guides_per_cell` from `obsm['guide_assignment']`). Mutate each `cNMF_<K>_<thresh>.h5mu` directly — CRT reads guide info from there.

---

## U-test Calibration

### Required parameters

- `--out_dir`, `--run_name`, `--mdata_guide_path`

### Key optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--components` | `[30,50,60,80,100,200,250,300]` | K values |
| `--sel_thresh` | `[0.4,0.8,2.0]` | Density thresholds |
| `--guide_annotation_key` | `['non-targeting']` | Non-targeting guide label |
| `--number_run` | `300` | Calibration iterations |
| `--number_guide` | `6` | Fake targeting guides per iteration |
| `--FDR_method` | `StoreyQ` | FDR correction method |

### SLURM resources

- Partition: `engreitz,owners`
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

- Partition: `engreitz,owners`
- CPUs: 10-20, Memory: 128-256G, Time: 2-6h
