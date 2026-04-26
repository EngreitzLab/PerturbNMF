# Advanced Parameters & CRT Calibration Reference

Read this file when the user needs Tier 2 advanced parameters or CRT calibration details.

---

## Tier 2 Advanced Parameters (Inference)

Present these after Tier 1 is collected. Ask: "Do you want to change any of these advanced parameters, or use the defaults?"

### Both sk-cNMF and torch-cNMF

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss` | `frobenius` | NMF loss function |
| `--nmf_seeds_path` | None | Custom NMF seeds file (text file with one integer per line) |
| `--num_gene` | `300` | Top genes for annotation |
| `--gene_names_key` | None | Column in adata.var with gene names for compiled results (e.g. `symbol`) |

### torch-cNMF only

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_jobs` | `-1` | Parallel jobs (-1 = all cores) |
| `--densify` | off | Densify sparse matrix before factorization |
| `--fp_precision` | `float` | `float` (32-bit) or `double` (64-bit) |
| `--alpha_usage` | `0.0` | Regularization for usage matrix (W) |
| `--alpha_spectra` | `0.0` | Regularization for spectra matrix (H) |
| `--l1_ratio_usage` | `0.0` | L1 vs L2 ratio for usage (0=L2, 1=L1) |
| `--l1_ratio_spectra` | `0.0` | L1 vs L2 ratio for spectra |
| `--minibatch_shuffle` | off | Shuffle cells in minibatch mode |
| `--remove_noncoding` | off | Remove non-coding genes |
| `--sk_cd_refit` | off | Use sklearn coordinate descent for refitting |

### torch-cNMF batch mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_max_epoch` | `500` | Max epochs for batch NMF |
| `--batch_hals_tol` | `0.05` | HALS tolerance |
| `--batch_hals_max_iter` | `200` | Max HALS inner iterations |

### torch-cNMF minibatch mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--minibatch_max_epoch` | `20` | Max epochs through data |
| `--minibatch_size` | `5000` | Minibatch size |
| `--minibatch_max_iter` | `200` | Max iterations per minibatch |
| `--minibatch_usage_tol` | `0.05` | Usage update tolerance |
| `--minibatch_spectra_tol` | `0.05` | Spectra update tolerance |

---

## CRT Calibration Details

### Step A: Determine guide data format

Check the h5mu structure to identify how guide/perturbation data is stored. Read the structure file if available:

```bash
cat <out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>_structure.txt
```

Look for these keys in the cNMF modality:
- **Standard keys** (CRT-compatible): `obsm['guide_assignment']`, `uns['guide_names']`, `uns['guide_targets']` → No preprocessing needed. Pass the h5mu or a guide h5ad directly as `--mdata_guide_path`.
- **Alternative keys** (common in KO/village experiments): `obsm['target_assignment']`, `uns['target_names']` → Preprocessing required. Use the `--preprocess_guide_h5mu` flag in `generate_slurm.py` to auto-inject a conversion step.

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
- `--mdata_guide_path`: Path to guide h5ad (from preprocessing) or h5mu with standard guide keys

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

**Redundancy warning**: Some datasets have multiple equivalent columns (e.g., `nCount_RNA` ≈ `n_counts`, `nFeature_RNA` ≈ `n_genes` ≈ `n_genes2`). Use only one from each group to avoid multicollinearity.

### Step E: Resource estimation for CRT

| Factor | Recommendation |
|--------|---------------|
| CPUs | `40` (CRT parallelizes permutations across cores) |
| Memory | `256G` for >100k cells, `128G` for <100k |
| Time | ~2h per K × sel_thresh × categorical condition. E.g., 1 K × 1 sel_thresh × 17 timepoints = 4-8h |
| Partition | `owners,engreitz,bigmem` |

### Step F: Generate SLURM script

If guide data preprocessing is needed (Step A), include the `--preprocess_guide_h5mu` flag:

```bash
python3 SKILL_DIR/scripts/generate_slurm.py \
  --stage crt-calibration \
  --job_name CRT_<run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus 40 --mem 256G --time 04:00:00 \
  --partition owners,engreitz,bigmem \
  --preprocess_guide_h5mu <out_dir>/<run_name>/Inference/adata/cNMF_<smallest_K>_<sel_thresh>.h5mu \
  --preprocess_guide_output <save_dir>/guide_data.h5ad \
  [--preprocess_assignment_key target_assignment] \
  [--preprocess_names_key target_names] \
  --log_dir <save_dir>/logs \
  --script_output_path <project_root>/Script/<run_name>_crt.sh \
  -- \
  --out_dir <out_dir> \
  --run_name <run_name> \
  --mdata_guide_path <save_dir>/guide_data.h5ad \
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

If no preprocessing is needed (standard guide keys), omit the `--preprocess_*` flags and point `--mdata_guide_path` directly to the guide data file.
