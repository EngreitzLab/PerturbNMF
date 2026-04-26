# Parameter Catalog

Complete parameter reference for all pipeline stages, extracted from argparse definitions.

**Note**: Parameter names with typos (e.g., `--run_complie_annotation` in sk-cNMF, `--top_enrichned_term`) are intentional -- they match the actual argparse definitions in the pipeline scripts and must be used as-is. The torch-cNMF version has fixed some of these (e.g., `--run_compile_annotation`).

---

## 1. sk-cNMF Inference

**Script**: `src/Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py`
**Conda**: `sk-cNMF`

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--counts_fn` | str | Path to input counts file (.h5ad or .h5mu) |
| `--output_directory` | str | Directory for all outputs |
| `--run_name` | str | Name for this cNMF run |
| `--species` | str | Species for gene annotation (`human`, `mouse`) |

### Core cNMF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--K` | int (nargs=\*) | `[30,50,60,80,100,200,250,300]` | K values (number of components) |
| `--numiter` | int | `10` | Number of NMF replicates |
| `--numhvgenes` | int | `5451` | Number of highly variable genes |
| `--sel_thresh` | float (nargs=\*) | `[2.0]` | Density thresholds for consensus filtering |
| `--seed` | int | `14` | Random seed |
| `--init` | str | `random` | NMF initialization (`random`, `nndsvd`, `nndsvda`, `nndsvdar`) |
| `--loss` | str | `frobenius` | NMF loss function |
| `--algo` | str | `mu` | NMF algorithm (`mu` or `cd`) |
| `--max_NMF_iter` | int | `500` | Maximum NMF iterations |
| `--tol` | float | `1e4` | Convergence tolerance (**WARNING**: likely a bug, should be `1e-4` -- see torch-cNMF) |

### Workflow Flags

| Flag | Description |
|------|-------------|
| `--run_factorize` | Run the NMF factorization step |
| `--run_refit` | Run combine, k_selection_plot, and consensus steps |
| `--run_complie_annotation` | Compile results and generate gene annotations (**note typo**: use as-is) |
| `--parallel_running` | Enable parallel processing for multiple K values |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--nmf_seeds_path` | str | None | Path to .npy file with custom NMF seeds |
| `--num_gene` | int | `300` | Top genes for annotation |
| `--gene_names_key` | str | None | Column in adata.var with gene names for compiled results (e.g. `symbol`) |

### Keys

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_key` | `rna` | RNA modality key in MuData |
| `--prog_key` | `cNMF` | cNMF modality key in MuData |
| `--categorical_key` | `sample` | Categorical variable key in obs |
| `--guide_names_key` | `guide_names` | Guide names key in uns |
| `--guide_targets_key` | `guide_targets` | Guide targets key in uns |
| `--guide_assignment_key` | `guide_assignment_key` | Guide assignment key in obsm (**NOTE**: default is the literal string `"guide_assignment_key"`, not `"guide_assignment"` -- this differs from torch-cNMF) |

---

## 2. torch-cNMF Inference

**Script**: `src/Inference/torch-cNMF/Slurm_Version/torch_cnmf_inference_pipeline.py`
**Conda**: `torch-cNMF`

torch-cNMF shares most parameters with sk-cNMF but has these important differences:
- Does NOT have `--max_NMF_iter` (use `--batch_max_epoch` instead)
- Does NOT have `--check_format`, `--guide_annotation_path`, or `--reference_gtf_path`
- `--K` default is `[5, 7, 10]` (smaller testing range)
- `--numhvgenes` default is `2000` (not `5451`)
- `--sel_thresh` default is `[2.0]`
- `--algo` default is `halsvar` (not `mu`)
- `--tol` default is `1e-4` (correct, unlike sk-cNMF's `1e4`)
- `--guide_assignment_key` default is `guide_assignment` (not `guide_assignment_key`)
- `--run_compile_annotation` has the correct spelling (no typo)
- "online" mode is now called "minibatch"; all `--online_*` parameters are now `--minibatch_*`
- `--batch_max_iter` is now `--batch_max_epoch`

### Required Parameters

Same as sk-cNMF: `--counts_fn`, `--output_directory`, `--run_name`, `--species`

### Core cNMF Parameters (shared with sk-cNMF, different defaults)

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `--K` | int (nargs=\*) | `[5, 7, 10]` | **Different from sk-cNMF** |
| `--numiter` | int | `10` | |
| `--numhvgenes` | int | `2000` | **Different from sk-cNMF** |
| `--sel_thresh` | float (nargs=\*) | `[2.0]` | |
| `--seed` | int | `14` | |
| `--init` | str | `random` | |
| `--loss` | str | `frobenius` | |
| `--algo` | str | `halsvar` | **Different from sk-cNMF** (options: `halsvar`, `mu`, `hals`, `bpp`) |
| `--tol` | float | `1e-4` | **Different from sk-cNMF** |

### Workflow Flags

| Flag | Description |
|------|-------------|
| `--run_factorize` | Run the factorization step |
| `--run_refit` | Run combine + k_selection + consensus |
| `--run_compile_annotation` | Compile results and gene annotation (**fixed spelling**, not `complie`) |
| `--parallel_running` | Enable parallel processing for multiple K values |

### Optional Parameters (shared with sk-cNMF)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--nmf_seeds_path` | str | None | Path to text file with custom NMF seeds (one integer per line) |
| `--num_gene` | int | `300` | Top genes for annotation |
| `--gene_names_key` | str | None | Column in adata.var with gene names for compiled results |
| `--tpm_fn` | str | None | Pre-computed TPM file path |
| `--genes_file` | str | None | File with gene list (overrides HVG selection) |

### Keys

Same as sk-cNMF except:
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--guide_assignment_key` | `guide_assignment` | **Different from sk-cNMF** (which defaults to `guide_assignment_key`) |

### Additional torch-cNMF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | str | `batch` | NMF mode: `batch`, `minibatch`, or `dataloader` |
| `--n_jobs` | int | `-1` | Parallel jobs (-1 = all cores) |
| `--use_gpu` | flag | False | Enable GPU acceleration |
| `--densify` | flag | False | Densify sparse matrix before factorization |

### Regularization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--alpha_usage` | float | `0.0` | Regularization for usage matrix |
| `--alpha_spectra` | float | `0.0` | Regularization for spectra matrix |
| `--l1_ratio_usage` | float | `0.0` | L1 ratio for usage (0=L2, 1=L1) |
| `--l1_ratio_spectra` | float | `0.0` | L1 ratio for spectra |

### Batch Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_max_epoch` | int | `500` | Max epochs for batch NMF |
| `--batch_hals_tol` | float | `0.05` | HALS tolerance in batch mode |
| `--batch_hals_max_iter` | int | `200` | Max HALS iterations |

### Minibatch Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--minibatch_max_epoch` | int | `20` | Max epochs through data |
| `--minibatch_size` | int | `5000` | Minibatch size |
| `--minibatch_max_iter` | int | `200` | Max iterations per minibatch |
| `--minibatch_usage_tol` | float | `0.05` | Usage update tolerance |
| `--minibatch_spectra_tol` | float | `0.05` | Spectra update tolerance |

### Other torch-cNMF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--fp_precision` | str | `float` | Floating point: `float` (32-bit) or `double` (64-bit) |
| `--minibatch_shuffle` | flag | False | Shuffle cells in minibatch mode |
| `--sk_cd_refit` | flag | False | Use sklearn coordinate descent for refitting |
| `--remove_noncoding` | flag | False | Remove non-coding genes by Ensembl prefix |
| `--ensembl_prefix` | str | `ENSG` | Prefix for identifying non-coding genes |
| `--gene_symbol_key` | str | `symbol` | Column in var with gene symbols |

---

## 3. Evaluation

**Script**: `src/Evaluation/Slurm_Version/cNMF_evaluation_pipeline.py`
**Conda**: `NMF_Benchmarking`

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `--out_dir` | str | Directory containing cNMF output |
| `--run_name` | str | cNMF run name |

### Test Flags

| Flag | Description |
|------|-------------|
| `--Perform_categorical` | Categorical association (Kruskal-Wallis + Dunn's test) |
| `--Perform_perturbation` | Perturbation association |
| `--Perform_geneset` | Gene set enrichment (Reactome + GO terms) |
| `--Perform_trait` | GWAS trait enrichment |
| `--Perform_explained_variance` | Explained variance per K |
| `--Perform_motif` | TF motif enrichment (WIP) |

### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--K` | int (nargs=\*) | `[30,50,60,80,100,200,250,300]` | K values to evaluate |
| `--sel_thresh` | float (nargs=\*) | `[0.4,0.8,2.0]` | Density thresholds |
| `--gwas_data_path` | str | None | Path to GWAS data (required only for trait enrichment) |
| `--X_normalized_path` | str | None | Normalized counts h5ad (needed for explained variance). Path pattern: `<out_dir>/<run_name>/Inference/cnmf_tmp/Inference.norm_counts.h5ad` (prefix is always `Inference`, not the run_name) |
| `--guide_annotation_path` | str | None | Guide annotation TSV |
| `--data_guide_path` | str | None | MuData with additional guide info |
| `--organism` | str | `human` | Species for enrichment |
| `--FDR_method` | str | `StoreyQ` | FDR method: `StoreyQ` or `BH` |
| `--n_top` | int | `300` | Top genes for enrichment tests |
| `--gene_names_key` | str | `symbol` | Column in data_guide["rna"].var with gene names |
| `--guide_annotation_key` | str (nargs=\*) | `["non-targeting"]` | Non-targeting guide identifiers (accepts multiple values) |

### Keys

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_key` | `rna` | RNA modality key |
| `--prog_key` | `cNMF` | cNMF modality key |
| `--categorical_key` | `sample` | Categorical variable key |
| `--guide_names_key` | `guide_names` | Guide names key |
| `--guide_targets_key` | `guide_targets` | Guide targets key |
| `--guide_assignment_key` | `guide_assignment` | Guide assignment key |

---

## 4. K-Selection Plot

**Script**: `src/Interpretation/Plotting/Slurm_Version/cNMF_k_selection.py`
**Conda**: `torch-cNMF`

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--output_directory` | str | -- | Yes | Directory with cNMF output |
| `--run_name` | str | -- | Yes | cNMF run name |
| `--save_folder_name` | str | -- | Yes | Where to save plots |
| `--eval_folder_name` | str | -- | Yes | Path to Eval results |
| `--K` | int (nargs=\*) | (from data) | No | K values |
| `--sel_threshs` | float (nargs=\*) | (from data) | No | Density thresholds |
| `--groupby` | str | `sample` | No | Grouping variable |
| `--pval` | float | `0.05` | No | P-value threshold |
| `--samples` | str (nargs=\*) | `['D0','sample_D1','sample_D2','sample_D3']` | No | Sample names |
| `--selected_k` | int | None | No | K value to highlight |
| `--go_file` | str | None | No | GO enrichment file pattern (use `{k}` placeholder) |
| `--geneset_file` | str | None | No | Geneset enrichment file pattern (use `{k}` placeholder) |
| `--trait_file` | str | None | No | Trait enrichment file pattern (use `{k}` placeholder) |
| `--term_col` | str | `Term` | No | Column name for term/pathway name |
| `--adjpval_col` | str | `Adjusted P-value` | No | Column name for adjusted p-value |
| `--perturbation_file` | str | None | No | Perturbation file pattern (use `{k}` and `{sample}`) |
| `--perturb_adjpval_col` | str | `adj_pval` | No | Perturbation adjusted p-value column |
| `--perturb_target_col` | str | `target_name` | No | Perturbation target column |
| `--perturb_log2fc_col` | str | `log2FC` | No | Perturbation log2FC column |
| `--variance_file` | str | None | No | Explained variance file pattern (use `{k}`) |
| `--variance_col` | str | `Total` | No | Variance column name |
| `--stability_file` | str | None | No | Pre-computed stability/error file (TSV or NPZ) |

---

## 5. Program Analysis Plot

**Script**: `src/Interpretation/Plotting/Slurm_Version/cNMF_program_analysis.py`
**Conda**: `NMF_Benchmarking`

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--mdata_path` | str | -- | Yes | Path to .h5mu file |
| `--perturb_path_base` | str | -- | Yes | Base path for perturbation results |
| `--GO_path` | str | -- | Yes | Path to GO enrichment results |
| `--pdf_save_path` | str | -- | Yes | Output directory for plots |
| `--file_to_dictionary` | str | None | No | Gene name mapping file |
| `--reference_gtf_path` | str | None | No | Reference GTF |
| `--tagert_col_name` | str | `program_name` | No | Target column in perturbation results |
| `--plot_col_name` | str | `target_name` | No | Gene column |
| `--log2fc_col` | str | `log2FC` | No | Log2FC column |
| `--top_program` | int | `10` | No | Top programs to display |
| `--top_enrichned_term` | int | `10` | No | Top GO terms per program |
| `--p_value` | float | `0.05` | No | Significance threshold |
| `--down_thred_log` | float | `-0.00` | No | Lower volcano threshold |
| `--up_thred_log` | float | `0.00` | No | Upper volcano threshold |
| `--figsize` | float (nargs=2) | `35 35` | No | Figure size |
| `--show` | flag | | No | Display interactively |
| `--PDF` | flag | | No | Save as PDF (else SVG) |
| `--square_plots` | flag | | No | Square aspect ratio |
| `--sample` | str (nargs=\*) | `['D0','sample_D1','sample_D2','sample_D3']` | No | Sample names |
| `--programs` | int (nargs=+) | None | No | Specific program numbers to plot (e.g. `4 5 6`). If omitted, all programs plotted |
| `--subsample_frac` | float | None | No | Fraction of cells to subsample for UMAP (e.g. `0.1` for 10%) |
| `--corr_matrix_path` | str | None | No | Base path for precomputed waterfall correlation matrices |

### Keys

`--data_key` (rna), `--prog_key` (cNMF), `--gene_name_key` (gene_names), `--categorical_key` (sample)

---

## 6. Perturbed Gene Analysis Plot

**Script**: `src/Interpretation/Plotting/Slurm_Version/cNMF_perturbed_gene_analysis.py`
**Conda**: `NMF_Benchmarking`

**Note**: Parameter names have been updated. Old name → new name mapping:
- `--file_to_dictionary` → `--ensembl_to_symbol_file`
- `--tagert_col_name` → `--perturb_target_col`
- `--plot_col_name` → `--perturb_program_col`
- `--log2fc_col` → `--perturb_log2fc_col`
- `--top_num` → `--top_corr_genes`
- `--top_program` → `--top_n_programs`
- `--p_value` → `--significance_threshold`
- `--down_thred_log` → `--volcano_log2fc_min`
- `--up_thred_log` → `--volcano_log2fc_max`
- `--pdf_save_path` → `--save_path`
- `--dot_size` → `--umap_dot_size`

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--mdata_path` | str | -- | Yes | Path to .h5mu file |
| `--perturb_path_base` | str | -- | Yes | Base path for perturbation results |
| `--save_path` | str | -- | Yes | Output directory for plots |
| `--ensembl_to_symbol_file` | str | None | No | Gene name mapping file |
| `--reference_gtf_path` | str | None | No | Reference GTF |
| `--perturb_target_col` | str | `target_name` | No | Target gene column in perturbation results |
| `--perturb_program_col` | str | `program_name` | No | Program column in perturbation results |
| `--perturb_log2fc_col` | str | `log2FC` | No | Log2FC column |
| `--top_corr_genes` | int | `5` | No | Top correlated genes per program |
| `--top_n_programs` | int | `10` | No | Top programs to display per gene |
| `--significance_threshold` | float | `0.05` | No | P-value threshold |
| `--volcano_log2fc_min` | float | `-0.00` | No | Lower volcano threshold |
| `--volcano_log2fc_max` | float | `0.00` | No | Upper volcano threshold |
| `--figsize` | float (nargs=2) | `35 35` | No | Figure size |
| `--show` | flag | | No | Display interactively |
| `--PDF` | flag | | No | Save as PDF (else SVG) |
| `--square_plots` | flag | | No | Square aspect ratio |
| `--n_processes` | int | `-1` | No | Parallel processes |
| `--umap_dot_size` | int | `10` | No | UMAP dot size |
| `--expressed_only` | flag | | No | Only plot expressed perturbed genes |
| `--sample` | str (nargs=\*) | `['D0','sample_D1','sample_D2','sample_D3']` | No | Sample names |
| `--gene_list_file` | str | None | No | File with gene names to process (one per line, overrides auto-detection) |
| `--subsample_frac` | float | None | No | Fraction of cells to subsample for UMAP |
| `--parallel` | flag | | No | Use fork-based multiprocessing (Linux only) |
| `--corr_matrix_path` | str | None | No | Directory for precomputed correlation matrices |
| `--control_target_name` | str | `non-targeting` | No | Name of non-targeting control in guide_targets |

### Keys

`--data_key` (rna), `--prog_key` (cNMF), `--gene_name_key` (gene_names), `--categorical_key` (sample)

---

## 7. U-test Calibration

**Script**: `src/Calibration/Slurm_version/U-test_perturbation_calibration/U-test_perturbation_calibration.py`
**Conda**: `NMF_Benchmarking`

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `--out_dir` | str | Directory containing cNMF output |
| `--run_name` | str | cNMF run name |
| `--mdata_guide_path` | str | Path to MuData with guide assignments |

### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--components` | int (nargs=\*) | `[30,50,60,80,100,200,250,300]` | K values |
| `--sel_thresh` | float (nargs=\*) | `[0.4,0.8,2.0]` | Density thresholds |
| `--guide_annotation_path` | str | None | Guide annotation TSV |
| `--guide_annotation_key` | str | `['non-targeting']` | Non-targeting guide label |
| `--reference_gtf_path` | str | None | Reference GTF |
| `--number_run` | int | `300` | Calibration iterations |
| `--number_guide` | int | `6` | Fake targeting guides per iteration |
| `--organism` | str | `human` | Species |
| `--FDR_method` | str | `StoreyQ` | FDR correction method |

### Workflow Flags

| Flag | Description |
|------|-------------|
| `--compute_real_perturbation_tests` | Run real perturbation tests |
| `--compute_fake_perturbation_tests` | Run calibration null distribution |
| `--visualizations` | Generate QQ and violin plots |
| `--check_format` | Validate format first |

### Keys

| Parameter | Default |
|-----------|---------|
| `--data_key` | `rna` |
| `--prog_key` | `cNMF` |
| `--categorical_key` | `sample` |
| `--guide_names_key` | `guide_names` |
| `--guide_targets_key` | `guide_targets` |
| `--guide_assignment_key` | `guide_assignment` |

---

## 8. CRT Calibration

**Script**: `src/Calibration/Slurm_version/CRT/CRT.py`
**Conda**: `programDE`

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `--out_dir` | str | Directory containing cNMF output |
| `--run_name` | str | cNMF run name |
| `--mdata_guide_path` | str | Path to MuData with guide assignments |

### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--components` | int (nargs=\*) | `[30,50,60,80,100,200,250,300]` | K values |
| `--sel_thresh` | float (nargs=\*) | `[0.4,0.8,2.0]` | Density thresholds |
| `--categorical_key` | str | `sample` | Sample/condition key |
| `--covariates` | str (nargs=\*) | None | Covariate keys in obs (used as-is) |
| `--log_covariates` | str (nargs=\*) | None | Covariate keys to log1p-transform |
| `--number_guide` | int | `6` | Fake targeting guides per iteration |
| `--number_permutations` | int | `1024` | CRT permutations |
| `--guide_annotation_key` | str (nargs=\*) | `non-targeting` | Non-targeting label (accepts multiple values) |
| `--FDR_method` | str | `BH` | `BH` or `StoreyQ` (choices enforced) |
| `--save_dir` | str | None | Custom save directory (default: `<out_dir>/<run_name>/Evaluation/<K>_<sel_thresh>/`) |

---

## 9. Matched Cell ProgramDE

**Script**: `src/Calibration/Slurm_version/Matched_cell_programDE/run_matching_de_batch.R`
**Utility**: `src/Calibration/Slurm_version/Matched_cell_programDE/de_testing_utils.R`
**Conda**: `programDE`

This is an R-based calibration method using propensity score matching with OLS regression and HC3 robust standard errors. It supports gene propagation (mapping program-level effects to genes via gene spectra) and direct gene-level DE testing.

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | str | Input directory |
| `--output_dir` | str | Output directory for results |
| `--cell_metadata` | str | Path to cells_metadata.tsv |
| `--cnmf_usages` | str | cNMF usages file (program loadings) |
| `--gene_perturbations_sparse` | str | Guide assignment file |
| `--gene_batch_file` | str | File with batch gene names to test (one per line) |
| `--perturbation_names` | str | File with all gene names to test (one per line) |
| `--condition` | str | Condition to test |
| `--batch_id` | str | Batch identifier for output filename |
| `--script_dir` | str | Directory containing de_testing_utils.R |

### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--test_type` | str | `genes` | What to test: `genes` or `nulls` |
| `--method` | str | `matching` | Statistical method |
| `--min_cells_per_gene` | int | `50` | Minimum cells required per gene |
| `--n_jobs` | int | `1` | Number of parallel jobs |
| `--matching_ratio` | int | `5` | Controls per treated cell |
| `--matching_method` | str | `nearest` | Matching algorithm |
| `--matching_calipers` | str | `""` | Calipers as covariate:width pairs |
| `--covariates` | str | `""` | Covariate column names (space-separated) |
| `--log_transform_covariates` | str | `""` | Covariates to log-transform |
| `--categorical_covariates` | str | `""` | Categorical covariates to convert to factors |
| `--methods` | str | `ols_log` | DE methods (space-separated: `ols_log`, `ols_pseudo`, `hurdle_marginal`) |
| `--save_matched_cells` | flag | False | Save matched cell indices |
| `--max_control_cells` | int | `20000` | Max control cells for matching |
| `--seed` | int | `42` | Random seed |

### Gene Propagation Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gene_spectra_tpm` | str | `""` | Path to gene spectra TPM file (W matrix); enables gene propagation |
| `--n_permutations` | int | `100` | Number of permutations for empirical bias correction |
| `--n_bootstrap` | int | `500` | Bootstrap iterations for SE estimation |
| `--bootstrap_ncpus` | int | `1` | Cores for bootstrap parallelization |

### Direct Gene-Level DE Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--de_level` | str | `programs` | DE level: `programs`, `genes`, or `both` |
| `--gene_expression` | str | `""` | Path to gene_expression_sparse.tsv for direct gene DE |
| `--response_gene_names` | str | `""` | Path to response_gene_names.txt |
