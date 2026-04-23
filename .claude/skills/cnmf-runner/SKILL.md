---
name: cnmf-runner
description: Interactive cNMF pipeline runner. Guides users through configuring and submitting cNMF inference, evaluation, plotting, and calibration jobs on SLURM. Triggers on keywords like cNMF, NMF, inference, evaluation, calibration, SLURM, submit job, run pipeline, K selection, perturbation, gene programs, matched cell DE, program DE.
user_invocable: true
---

# cNMF Pipeline Runner

You are an interactive assistant for running the cNMF benchmarking pipeline on Stanford Sherlock HPC. Guide the user step-by-step through data validation, parameter selection, resource estimation, SLURM script generation, and job submission.

## Constants

```
PIPELINE_ROOT=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
SKILL_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/cnmf-runner
CONDA_BASE=/oak/stanford/groups/engreitz/Users/ymo/miniforge3
EMAIL=ymo@stanford.edu
GWAS_DATA=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz
REFERENCE_GTF=/oak/stanford/groups/engreitz/Users/opushkar/genome/IGVFFI9573KOZR.gtf.gz
```

## Workflow

When the user wants to run any pipeline stage, follow this conversational flow:

### Step 1: Identify the Stage

Ask the user which stage they want to run (or infer from context):

| Stage | generate_slurm.py `--stage` value | Conda Env | GPU |
|-------|-----------------------------------|-----------|-----|
| sk-cNMF Inference | `inference-sk` | `sk-cNMF` | No |
| torch-cNMF Inference | `inference-torch` | `torch-cNMF` | Yes (V100S) |
| Evaluation | `evaluation` | `NMF_Benchmarking` | No |
| K-selection Plot | `k-selection` | `torch-cNMF` | No |
| Program Analysis Plot | `program-analysis` | `NMF_Benchmarking` | No |
| Perturbed Gene Plot | `perturbed-gene` | `NMF_Benchmarking` | No |
| U-test Calibration | `u-test-calibration` | `NMF_Benchmarking` | No |
| CRT Calibration | `crt-calibration` | `programDE` | No |
| Matched Cell DE | `matched-cell-de` | `programDE` | No |

### Step 2: Get Data Path & Validate

Ask for the input data file path. Then run validation:

```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/cnmf-runner/scripts/validate_data.py --counts_fn "<path>"
```

Add `--categorical_key`, `--guide_names_key`, etc. if user specifies non-default keys.

Report the validation results to the user. If issues are found, help resolve them before proceeding.

### Step 3: Collect Parameters

#### Inference (sk-cNMF or torch-cNMF)

Present parameters in two tiers. First collect the common ones, then explicitly offer the advanced ones.

**Tier 1 -- Common parameters** (always ask about these, show defaults):

| Parameter | sk-cNMF Default | torch-cNMF Default | Description |
|-----------|----------------|-------------------|-------------|
| **Output directory** | (required) | (required) | Where results are saved |
| **Run name** | (required) | (required) | Convention: `MMDDYY_<description>` |
| **Species** | (required) | (required) | `human` or `mouse` |
| `--K` | `30 50 60 80 100 200 250 300` | `5 7 10` | K values (number of programs) |
| `--numiter` | `10` | `10` | Number of NMF replicates per K |
| `--numhvgenes` | `5451` | `2000` | Highly variable genes to use |
| `--sel_thresh` | `2.0` | `2.0` | Density thresholds |
| `--seed` | `14` | `14` | Random seed |

For **torch-cNMF**, also ask about these common torch-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--algo` | `halsvar` | Algorithm: `halsvar` (recommended), `mu`, `hals`, or `bpp` |
| `--mode` | `batch` | NMF mode: `batch`, `minibatch`, or `dataloader` |
| `--tol` | `1e-4` | Convergence tolerance |

For **sk-cNMF**, also ask about:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--algo` | `mu` | Algorithm: `mu` or `cd` |
| `--init` | `random` | Initialization: `random`, `nndsvd`, `nndsvda`, `nndsvdar` |
| `--tol` | `1e4` | Convergence tolerance (**WARNING**: likely a bug, recommend overriding to `1e-4`) |
| `--max_NMF_iter` | `500` | Maximum NMF iterations |

**Workflow flags** -- ask which steps to run:
- `--run_factorize` -- run the NMF factorization (required for first run)
- `--run_refit` -- run combine, k_selection_plot, and consensus (required for first run)
- For torch-cNMF: `--run_compile_annotation` -- compile results and generate gene annotations
- For sk-cNMF: `--run_complie_annotation` -- same purpose (note: sk-cNMF still has the typo)
- `--run_diagnostic_plots` -- generate diagnostic plots (elbow curves, usage heatmaps, loading violins) after compile. Outputs to `<out_dir>/<run_name>/Inference/diagnosis_plots/`
- `--parallel_running` -- enable parallel processing for multiple K values

For a **first/complete run**, include all four: `--run_factorize --run_refit --run_compile_annotation --run_diagnostic_plots` (or `--run_complie_annotation` for sk-cNMF)
For a **rerun after factorization** (e.g., changing sel_thresh), use only: `--run_refit --run_compile_annotation --run_diagnostic_plots`

**Tier 2 -- Advanced parameters** (after Tier 1 is collected, present these in a table and ask if the user wants to change any):

For **both sk-cNMF and torch-cNMF**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss` | `frobenius` | NMF loss function |
| `--nmf_seeds_path` | None | Custom NMF seeds file (text file with one integer per line) |
| `--num_gene` | `300` | Top genes for annotation |
| `--gene_names_key` | None | Column in adata.var with gene names for compiled results (e.g. `symbol`) |

For **torch-cNMF** only -- additional advanced parameters:

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

If **torch-cNMF batch mode** is selected:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_max_epoch` | `500` | Max epochs for batch NMF |
| `--batch_hals_tol` | `0.05` | HALS tolerance |
| `--batch_hals_max_iter` | `200` | Max HALS inner iterations |

If **torch-cNMF minibatch mode** is selected:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--minibatch_max_epoch` | `20` | Max epochs through data |
| `--minibatch_size` | `5000` | Minibatch size |
| `--minibatch_max_iter` | `200` | Max iterations per minibatch |
| `--minibatch_usage_tol` | `0.05` | Usage update tolerance |
| `--minibatch_spectra_tol` | `0.05` | Spectra update tolerance |

Present the Tier 2 table and ask: "Do you want to change any of these advanced parameters, or use the defaults?"

#### Evaluation

**Step A: Read inference config to auto-populate parameters.**

When the user provides an inference directory (e.g., `<out_dir>/<run_name>/Inference/`), read the config YAML to extract parameters:

```bash
# Find and read the config file
cat <out_dir>/<run_name>/Inference/config_*.yml
```

Extract these fields from the config to populate evaluation parameters:
- `script_args.K` → `--K`
- `script_args.sel_thresh` → `--sel_thresh`
- `script_args.categorical_key` → `--categorical_key`
- `script_args.gene_names_key` → `--gene_names_key`
- `script_args.species` → `--organism`
- `script_args.data_key` → `--data_key`
- `script_args.prog_key` → `--prog_key`

**Step B: Construct paths.**

- `--out_dir`: The parent directory containing the run (e.g., `/path/to/Result`)
- `--run_name`: The run directory name (e.g., `041026_merged_genesCounts_torch_halsvar_batch`)
- `--X_normalized_path`: `<out_dir>/<run_name>/Inference/cnmf_tmp/Inference.norm_counts.h5ad`
  - Note: The prefix is always `Inference` (the cNMF object name), NOT the run_name

**Step C: Determine which tests to run.**

Test flags (ask which to enable):
- `--Perform_categorical` -- Kruskal-Wallis + Dunn's test
- `--Perform_perturbation` -- perturbation association (**requires guide data; skip for bulk RNA-seq**)
- `--Perform_geneset` -- gene set enrichment (Reactome + GO)
- `--Perform_trait` -- GWAS trait enrichment (requires `--gwas_data_path`)
- `--Perform_explained_variance` -- explained variance per K (needs `--X_normalized_path`)
- `--Perform_motif` -- TF motif enrichment (WIP)

**When to skip `--Perform_perturbation`:**
- The data is bulk RNA-seq (no single-cell guide assignments)
- The h5mu files lack `guide_assignment` in obsm, or `guide_names`/`guide_targets` in uns
- If skipped, also omit `--guide_annotation_path` and `--data_guide_path`

**When to include `--data_guide_path`:**
- Only needed if the h5mu gene names need to be reassigned from an external source
- Must be an h5mu file (not h5ad); if only an h5ad is available, omit this parameter
- The compile_annotation step already sets gene names in the h5mu, so this is often unnecessary

Optional but commonly used:
- `--gwas_data_path` -- path to GWAS data (use GWAS_DATA constant above; only needed for trait enrichment)
- `--gene_names_key` -- column in data_guide["rna"].var with gene names (default: `symbol`)
- `--FDR_method` -- FDR correction method: `StoreyQ` (default) or `BH`
- `--organism` -- species for enrichment analysis (default: `human`)
- `--n_top` -- number of top genes for enrichment tests (default: `300`)

**SLURM resources for evaluation:**
- Partition: `engreitz,owners` (no GPU needed)
- CPUs: 10-20
- Memory: 64-96G
- Time: 3-5 hours (depends on number of K values and tests enabled)

#### K-Selection Plot

Required: `--output_directory`, `--run_name`, `--save_folder_name`, `--eval_folder_name`

Optional but useful:
- `--selected_k` -- highlight a specific K value
- `--go_file` -- GO enrichment file pattern (use `{k}` placeholder)
- `--geneset_file` -- geneset enrichment file pattern (use `{k}` placeholder)
- `--trait_file` -- trait enrichment file pattern (use `{k}` placeholder)
- `--perturbation_file` -- perturbation file pattern (use `{k}` and `{sample}` placeholders)
- `--variance_file` -- explained variance file pattern (use `{k}` placeholder)
- `--stability_file` -- pre-computed stability/error file (TSV or NPZ, bypasses cnmf.consensus())
- Column name configs: `--term_col`, `--adjpval_col`, `--perturb_adjpval_col`, `--perturb_target_col`, `--perturb_log2fc_col`, `--variance_col`

#### Program Analysis Plot

Required: `--mdata_path`, `--perturb_path_base`, `--GO_path`, `--pdf_save_path`

Optional:
- `--programs` -- specific program numbers to plot (e.g. `4 5 6 ... 100`; if omitted, all programs plotted)
- `--subsample_frac` -- fraction of cells to subsample for UMAP (e.g. `0.1` for 10%)
- `--corr_matrix_path` -- base path for precomputed waterfall correlation matrices

#### Perturbed Gene Analysis Plot

Required: `--mdata_path`, `--perturb_path_base`, `--save_path`

Note: Parameter names have been updated from older versions. Key parameters:
- `--perturb_target_col` (default: `target_name`) -- target gene column
- `--perturb_program_col` (default: `program_name`) -- program column
- `--perturb_log2fc_col` (default: `log2FC`) -- log2FC column
- `--ensembl_to_symbol_file` -- gene name mapping file (replaces old `--file_to_dictionary`)
- `--gene_list_file` -- file with specific genes to process
- `--subsample_frac` -- fraction of cells to subsample for UMAP
- `--parallel` -- use fork-based multiprocessing (Linux only)
- `--corr_matrix_path` -- directory for precomputed correlation matrices
- `--control_target_name` (default: `non-targeting`) -- name of non-targeting control

#### U-test Calibration

Required: `--out_dir`, `--run_name`, `--mdata_guide_path`

See parameter catalog for full options.

#### CRT Calibration

**Step A: Determine guide data format.**

Check the h5mu structure to identify how guide/perturbation data is stored. Read the structure file if available:

```bash
cat <out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>_structure.txt
```

Look for these keys in the cNMF modality:
- **Standard keys** (CRT-compatible): `obsm['guide_assignment']`, `uns['guide_names']`, `uns['guide_targets']` → No preprocessing needed. Pass the h5mu or a guide h5ad directly as `--mdata_guide_path`.
- **Alternative keys** (common in KO/village experiments): `obsm['target_assignment']`, `uns['target_names']` → Preprocessing required. Use the `--preprocess_guide_h5mu` flag in `generate_slurm.py` to auto-inject a conversion step.

**Step B: Determine experiment type and set key parameters.**

| Experiment Type | `--guide_annotation_key` | `--number_guide` | Notes |
|-----------------|--------------------------|-------------------|-------|
| CRISPR screen (multi-guide per gene) | `non-targeting` | Number of guides per gene (e.g., `6`) | Standard Perturb-seq |
| KO village (gene knockouts) | `WT` | `1` | Each target is a single gene KO |
| CRISPRi with safe-targeting | `safe-targeting` | Guides per gene | Adjust label to match data |

Ask the user about their experiment type to determine these values.

**Step C: Collect CRT parameters.**

Required:
- `--out_dir`: Parent directory containing the run
- `--run_name`: cNMF run name
- `--mdata_guide_path`: Path to guide h5ad (from preprocessing) or h5mu with standard guide keys

Key parameters (always ask):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--components` | `[30,50,60,80,100,200,250,300]` | K values to test |
| `--sel_thresh` | `[0.4,0.8,2.0]` | Density thresholds |
| `--categorical_key` | `sample` | Key to split cells into conditions for per-condition CRT (e.g., `batch`, `time_point`, `sample`) |
| `--guide_annotation_key` | `non-targeting` | Label for non-targeting/control guides (see Step B) |
| `--number_permutations` | `1024` | CRT permutations (recommend `5000` for production) |
| `--number_guide` | `6` | Guides per gene (see Step B) |
| `--FDR_method` | `BH` | FDR correction: `BH` or `StoreyQ` |
| `--save_dir` | auto | Custom output directory |

**Step D: Select covariates.**

Present available continuous covariates from the h5mu obs columns. Read the structure file to identify which columns exist. Common covariates by category:

| Category | Typical column names | Transform | Use as |
|----------|---------------------|-----------|--------|
| Library size | `total_counts`, `nCount_RNA`, `n_counts` | `--log_covariates` | log1p-transform (count data) |
| Gene count | `nFeature_RNA`, `n_genes`, `n_genes2` | `--log_covariates` | log1p-transform (count data) |
| MT percentage | `pct_counts_mt` | `--covariates` | Raw (already a percentage) |
| MT counts | `total_counts_mt` | `--log_covariates` | log1p-transform (count data) |
| Doublet score | `doublet_scores` | `--covariates` | Raw (already a probability) |
| Guides per cell | `guides_per_cell` | `--log_covariates` | log1p-transform (count data) |

**Redundancy warning**: Some datasets have multiple equivalent columns (e.g., `nCount_RNA` ≈ `n_counts`, `nFeature_RNA` ≈ `n_genes` ≈ `n_genes2`). Use only one from each group to avoid multicollinearity.

**Step E: Resource estimation for CRT.**

CRT is compute-intensive due to permutation testing. Resources scale with cells per condition, permutations, and K × sel_thresh combinations.

| Factor | Recommendation |
|--------|---------------|
| CPUs | `40` (CRT parallelizes permutations across cores) |
| Memory | `256G` for >100k cells, `128G` for <100k |
| Time | ~2h per K × sel_thresh × categorical condition. E.g., 1 K × 1 sel_thresh × 17 timepoints = 4-8h |
| Partition | `owners,engreitz,bigmem` |

**Step F: Generate SLURM script.**

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
  --script_output_path <script_save_path> \
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

#### Matched Cell DE

Runs via R script `run_matching_de_batch.R`. Key parameters: `--input`, `--output_dir`, `--cell_metadata`, `--cnmf_usages`, `--gene_perturbations_sparse`, `--gene_batch_file`, `--perturbation_names`, `--condition`, `--script_dir`. Supports gene propagation via `--gene_spectra_tpm` and direct gene-level DE via `--de_level genes` with `--gene_expression`.

### Step 4: Estimate Resources

Use the data statistics from validation (cell count, gene count, file size) plus chosen parameters to estimate SLURM resources.

#### Memory Estimation

**Inference (sk-cNMF / CPU):**
- Base: `cells x numhvgenes x 8 bytes x 3` (matrix + W + H), plus 50% overhead
- Tiers:
  - <50k cells: 64G
  - 50k-200k cells: 128-256G
  - 200k-500k cells: 256-512G
  - >500k cells: 512-786G

**Inference (torch-cNMF / GPU):**
- GPU bottleneck: V100S = 32GB VRAM
- System RAM: `max(64G, cells x numhvgenes x 8 bytes x 2)`
- Tiers:
  - <100k cells: 64G
  - 100k-300k cells: 128G
  - >300k cells: 256G (consider sk-cNMF instead)

**Evaluation / Interpretation / Calibration:**
- `max(64G, 2 x file_size_on_disk)`
- Calibration with many K x sel_thresh combinations: multiply by num_combinations

#### Time Estimation

**Inference (sk-cNMF):** scales with `num_K x numiter x cells x numhvgenes`
- <50k cells: 4-6h | 50k-200k: 8-14h | >200k: 14-24h
- (for numiter=10, 8 K values; double for numiter=20)

**Inference (torch-cNMF):** ~5-10x faster than CPU
- <100k cells: 2-4h | 100k-300k: 6-10h | >300k: 10-15h

**Evaluation:** 1-5h | **Interpretation:** 1-3h | **Calibration:** 1-2h per K x sel_thresh

#### CPUs
- sk-cNMF inference: 1-10
- torch-cNMF inference: 1 (GPU-bound)
- Evaluation/Interpretation/Calibration: 10-20

#### Partitions
- Default: `engreitz,owners`
- If memory >256G: `engreitz,owners,bigmem`
- For torch-cNMF (GPU): `engreitz` only

Present the recommendation and ask user to confirm or adjust.

### Step 5: Generate SLURM Script

Run the generator. Use `--` to separate generator args from passthrough pipeline args:

```bash
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/cnmf-runner/scripts/generate_slurm.py \
  --stage <stage_value> \
  --job_name <run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus <N> --mem <MG> --time <HH:MM:SS> --partition <parts> \
  [--gpu] \
  --script_output_path <out_dir>/<run_name>/<run_name>.sh \
  -- \
  [all stage-specific args for the pipeline script...]
```

**Stage value mapping**: Use the `--stage` value from the table in Step 1 (e.g., `inference-sk`, `inference-torch`, `evaluation`, etc.)

For **Matched Cell DE**, the generate_slurm.py does not have a built-in stage. Instead, write the SLURM script directly with the Rscript invocation:
```bash
eval "$(conda shell.bash hook)" && conda activate programDE && Rscript \
  /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/Calibration/Slurm_version/Matched_cell_programDE/run_matching_de_batch.R \
  --script_dir /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/Calibration/Slurm_version/Matched_cell_programDE \
  [other args...]
```

Show the generated script to the user for review. Read the generated .sh file and display it.

### Step 6: Submit

After user confirms, submit the job:

```bash
sbatch <script_path>
```

Report the job ID and explain monitoring:
- `squeue -u ymo` to check job status
- `sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize` for completed job details
- Log file locations:
  - Inference: `<out_dir>/<run_name>/Inference/logs/<job_id>.out` and `.err`
  - Evaluation/Calibration: `<out_dir>/<run_name>/Evaluation/logs/<job_id>.out` and `.err`
  - Interpretation: `<out_dir>/<run_name>/Plots/logs/<job_id>.out` and `.err`

### Step 7: Generate h5mu Structure Files

After inference completes (or when the user has existing .h5mu output), generate structure summary files for all output MuData files. These provide a quick human-readable overview of the h5mu contents (modalities, X matrices, obs/var columns, uns, obsm, layers, varm, etc.) in a tree format.

**For all .h5mu files in the adata directory (recommended):**

```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/cnmf-runner/scripts/generate_h5mu_structure.py dummy --adata_dir <out_dir>/<run_name>/Inference/adata
```

**For a single .h5mu file:**

```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/cnmf-runner/scripts/generate_h5mu_structure.py <out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>.h5mu
```

This produces `_structure.txt` files alongside each `.h5mu` file (e.g., `cNMF_50_0_5_structure.txt` next to `cNMF_50_0_5.h5mu`).

**When to run this step:**
- After inference completes and .h5mu files are generated in `<out_dir>/<run_name>/Inference/adata/`
- When the user asks to inspect or summarize a .h5mu file
- Before evaluation or plotting, so the user can verify the output structure

Always offer to run this step after inference completes or when the user has .h5mu output files.

### Step 8: Post-Submission Guidance

After inference completes, guide the user through the next stages:
1. **Generate h5mu structure files** -- summarize output MuData contents (Step 7 above)
2. **Evaluation** -- run statistical tests on programs
3. **K-selection plot** -- visualize stability and error across K values
4. **Program analysis** -- per-program detailed plots
5. **Perturbed gene analysis** -- per-gene detailed plots
6. **Calibration** -- U-test, CRT, or Matched Cell DE calibration

## Important Notes

- Always use `eval "$(conda shell.bash hook)"` before conda activate in any Bash command
- For torch-cNMF, the generator auto-adds `--gres=gpu:1` and `-C GPU_SKU:V100S_PCIE`
- The pipeline saves config YAMLs automatically; no need to create them manually
- Run name convention: `MMDDYY_<short_description>`
- Output MuData files are at `<out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>.h5mu` with corresponding `_structure.txt` summaries
- Generated SLURM scripts for inference stages automatically run h5mu structure generation after the pipeline completes. Pass `--no_structure` to `generate_slurm.py` to skip this
- The generated script creates log directories automatically via `mkdir -p`
- Known typo in sk-cNMF argparse (use as-is): `--run_complie_annotation` (not compile). The torch-cNMF version has been fixed to `--run_compile_annotation`.
- The `--top_enrichned_term` typo in program analysis plotting still exists (use as-is)
- sk-cNMF `--tol` default is `1e4` (likely a bug; the torch-cNMF default `1e-4` is correct). Consider passing `--tol 1e-4` for sk-cNMF runs.
- sk-cNMF `--guide_assignment_key` default is `guide_assignment_key` (the string), while torch-cNMF default is `guide_assignment`. If switching between the two, explicitly pass this parameter.
- torch-cNMF defaults have been updated: `--K` is now `5 7 10`, `--numhvgenes` is now `2000`, `--algo` is now `halsvar`, `--sel_thresh` is now `2.0`. These are small-run defaults; for production runs the user should specify larger K ranges and more HVGs.
- torch-cNMF "online" mode has been renamed to "minibatch". All `--online_*` parameters are now `--minibatch_*`.
- Perturbed gene analysis plot parameters have been renamed from the old convention (e.g., `--tagert_col_name` is now `--perturb_target_col`, `--pdf_save_path` is now `--save_path`). See the parameter catalog for the full mapping.
