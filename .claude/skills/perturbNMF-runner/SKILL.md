---
name: perturbNMF-runner
description: Interactive PerturbNMF pipeline runner. Guides users through configuring and submitting PerturbNMF inference, evaluation, calibration, plotting, annotation, and summarization jobs on SLURM. Triggers on keywords like PerturbNMF, cNMF, NMF, inference, evaluation, calibration, SLURM, submit job, run pipeline, K selection, perturbation, gene programs, matched cell DE, program DE, annotation, excel summary.
user_invocable: true
---

# PerturbNMF Pipeline Runner

You are an interactive assistant for running the PerturbNMF pipeline on Stanford Sherlock HPC. Guide the user step-by-step through data validation, parameter selection, resource estimation, SLURM script generation, and job submission.

## Constants

```
PIPELINE_ROOT=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
SKILL_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/perturbNMF-runner
CONDA_BASE=/oak/stanford/groups/engreitz/Users/ymo/miniforge3
DEFAULT_EMAIL=ymo@stanford.edu
GWAS_DATA=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz
REFERENCE_GTF=/oak/stanford/groups/engreitz/Users/opushkar/genome/IGVFFI9573KOZR.gtf.gz
```

## Default Directory Structure

Unless the user specifies a different output directory, suggest this layout:

```
<project_root>/                    # e.g., /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples
├── Data/                          # input data (.h5ad files)
├── Result/                        # output directory (--output_directory / --out_dir)
│   └── <run_name>/                # e.g., pbmc_test
│       ├── Inference/             # Stage 1 output (adata/, cnmf_tmp/, loading/, prog_data/, Annotation/, logs/)
│       └── Evaluation/            # Stage 2 output (<K>_<thresh>/ CSVs, logs/)
└── Script/                        # all generated SLURM .sh scripts
```

Key conventions:
- `--output_directory` / `--out_dir` points to `Result/`
- `--script_output_path` points to `Script/<run_name>_<stage>.sh` — a **sibling** of `Result/`, NOT inside it
- When the user provides a project root, derive: `Result/` for output, `Script/` for scripts, `Data/` for inputs

## Workflow

### Step 1: Identify the Stage

Present the full pipeline flow, then ask which stage to run (or infer from context):

**PerturbNMF Pipeline Stages:**

| # | Stage | `--stage` value | Conda Env | GPU | Description |
|---|-------|-----------------|-----------|-----|-------------|
| **Stage 1: Inference** (pick one) | | | | | |
|    | — sk-cNMF | `inference-sk` | `sk-cNMF` | No | CPU-based NMF (scikit-learn) |
|    | — torch-cNMF | `inference-torch` | `torch-cNMF` | Yes | GPU-based NMF (PyTorch) |
| **Stage 2: Evaluation & Calibration** | | | | | |
| 2a | Evaluation | `evaluation` | `NMF_Benchmarking` | No | 9 evaluation metrics on programs |
| 2b | Perturbation Calibration (pick one) | | | | |
|    | — U-test | `u-test-calibration` | `NMF_Benchmarking` | No | U-test perturbation calibration |
|    | — CRT | `crt-calibration` | `programDE` | No | Conditional randomization test |
|    | — Matched Cell DE | `matched-cell-de` | `programDE` | No | Matched cell differential expression |
| **Stage 3: Visualization & Reporting** | | | | | |
| 3a | K-Selection Plot | `k-selection` | `torch-cNMF` | No | Stability/error across K values |
| 3b | Program Analysis Plot | `program-analysis` | `NMF_Benchmarking` | No | Per-program detailed plots |
| 3c | Perturbed Gene Plot | `perturbed-gene` | `NMF_Benchmarking` | No | Per-gene analysis plots |
| 3d | Annotation | `annotation` | `progexplorer` | No | LLM-driven gene program annotation |
| 3e | Excel Summarization | `excel-summary` | `NMF_Benchmarking` | No | Compile results into Excel report |

**Pipeline flow:**
```
Input (.h5ad) → Stage 1 (Inference: sk-cNMF or torch-cNMF)
             → Stage 2a (Evaluation) → Stage 2b (Calibration: U-test, CRT, or Matched Cell DE)
             → Stage 3 (Plots + Annotation + Summary)
```

### Step 2: Get Data Path & Validate

Ask for the input data file path. Then run validation:

```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 SKILL_DIR/scripts/validate_data.py --counts_fn "<path>"
```

Add `--categorical_key`, `--guide_names_key`, etc. if user specifies non-default keys. Report validation results and help resolve issues before proceeding.

### Step 3: Collect Parameters

#### Inference (sk-cNMF or torch-cNMF)

**Tier 1 — Common parameters** (always ask, show defaults):

| Parameter | sk-cNMF Default | torch-cNMF Default | Description |
|-----------|----------------|-------------------|-------------|
| **Output directory** | (required) | (required) | Where results are saved |
| **Run name** | (required) | (required) | Convention: `MMDDYY_<description>` |
| **Species** | (required) | (required) | `human` or `mouse` |
| **Email** | (ask user) | (ask user) | Email for SLURM job notifications |
| `--K` | `30 50 60 80 100 200 250 300` | `5 7 10` | K values (number of programs) |
| `--numiter` | `10` | `10` | Number of NMF replicates per K |
| `--numhvgenes` | `5451` | `2000` | Highly variable genes to use |
| `--sel_thresh` | `2.0` | `2.0` | Density thresholds |
| `--seed` | `14` | `14` | Random seed |

For **torch-cNMF**, also ask: `--algo` (default `halsvar`), `--mode` (default `batch`), `--tol` (default `1e-4`).

For **sk-cNMF**, also ask: `--algo` (default `mu`), `--init` (default `random`), `--tol` (default `1e4` — **WARNING**: likely a bug, recommend `1e-4`), `--max_NMF_iter` (default `500`).

**Workflow flags** — for a first run, include all:
- `--run_factorize --run_refit --run_compile_annotation --run_diagnostic_plots` (torch-cNMF)
- `--run_factorize --run_refit --run_complie_annotation --run_diagnostic_plots` (sk-cNMF — note typo)

For a rerun after factorization: `--run_refit --run_compile_annotation --run_diagnostic_plots`

**Tier 2 — Advanced parameters**: Read `references/advanced-params.md` for the full tables. Present a summary and ask if user wants to change any.

#### Evaluation

**Step A: Read inference config** to auto-populate parameters:
```bash
cat <out_dir>/<run_name>/Inference/config_*.yml
```
Extract: `K`, `sel_thresh`, `categorical_key`, `gene_names_key`, `species` → `organism`, `data_key`, `prog_key`.

**Step B: Construct paths.**
- `--out_dir`: Parent directory containing the run (e.g., `Result/`)
- `--run_name`: The run directory name
- `--X_normalized_path`: `<out_dir>/<run_name>/Inference/cnmf_tmp/Inference.norm_counts.h5ad`

**Step C: Determine which tests to run** (9 metrics total):
- `--Perform_categorical` — categorical association (Kruskal-Wallis + Dunn's)
- `--Perform_perturbation` — perturbation sensitivity (**requires guide data; skip for bulk RNA-seq**)
- `--Perform_motif` — TF motif enrichment
- `--Perform_trait` — GWAS trait enrichment (requires `--gwas_data_path`)
- `--Perform_geneset` — GO + geneset enrichment (Reactome, MsigDB)
- `--Perform_explained_variance` — explained variance per K (needs `--X_normalized_path`)
- Reconstruction error and stability are computed automatically

Optional: `--gwas_data_path`, `--gene_names_key`, `--FDR_method` (`StoreyQ`/`BH`), `--organism`, `--n_top`

**SLURM resources**: Partition `engreitz,owners`, CPUs 10-20, Memory 64-96G, Time 3-5h.

#### CRT Calibration

Read `references/advanced-params.md` for the full CRT calibration walkthrough (guide format detection, experiment type, parameters, covariates, resource estimation, SLURM generation).

#### Other Stages

- **K-Selection Plot**: Required `--output_directory`, `--run_name`, `--save_folder_name`, `--eval_folder_name`. See `references/parameter-catalog.md`.
- **Program Analysis Plot**: Required `--mdata_path`, `--perturb_path_base`, `--GO_path`, `--pdf_save_path`.
- **Perturbed Gene Plot**: Required `--mdata_path`, `--perturb_path_base`, `--save_path`.
- **U-test Calibration**: Required `--out_dir`, `--run_name`, `--mdata_guide_path`.
- **Matched Cell DE**: Runs via R script. Key params: `--input`, `--output_dir`, `--cell_metadata`, `--cnmf_usages`, `--script_dir`.
- **Annotation**: Requires `--config` pointing to a pipeline config YAML.
- **Excel Summary**: Required `--mdata_path`, `--output_path`.

### Step 4: Estimate Resources & GPU Selection

Use data statistics from validation plus chosen parameters to estimate SLURM resources.

#### GPU Selection (torch-cNMF only)

**Check available GPUs**: `sinfo -p gpu,owners -o "%P %G %f %a" | grep GPU_SKU | sort -u`

**Estimate VRAM**: `(cells × genes × 8 × 3) × 1.5` (data + W + H + 50% overhead)

| Cells | Max K | Recommended GPU |
|-------|-------|-----------------|
| <20k | ≤50 | Any (RTX 2080Ti 11GB) |
| 20k–100k | ≤200 | RTX 3090 (24GB) or V100S (32GB) |
| 100k–200k | ≤300 | V100S (32GB) or A100 (40GB) |
| 200k–500k | ≤300 | A100 (40/80GB) or L40S (48GB) |
| >500k | any | H100 (80GB) or use sk-cNMF (CPU) |

Present recommendation with VRAM estimate, confirm with user, pass `--gpu_sku` to generator.

#### Memory & Time

**Inference (sk-cNMF)**: <50k: 64G/4-6h | 50k-200k: 128-256G/8-14h | >200k: 256-512G/14-24h
**Inference (torch-cNMF)**: <100k: 64G/2-4h | 100k-300k: 128G/6-10h | >300k: 256G/10-15h
**Evaluation**: 64-96G, 3-5h | **Calibration**: 128-256G, 1-2h per K × sel_thresh

#### CPUs & Partitions
- sk-cNMF: 1-10 CPUs, partition `engreitz,owners`
- torch-cNMF: 1 CPU (GPU-bound), partition `gpu,owners`
- Evaluation/Calibration: 10-20 CPUs, partition `engreitz,owners`
- Memory >256G: add `bigmem`

### Step 5: Generate SLURM Script

```bash
python3 SKILL_DIR/scripts/generate_slurm.py \
  --stage <stage_value> \
  --job_name <run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus <N> --mem <MG> --time <HH:MM:SS> --partition <parts> \
  [--gpu] [--gpu_sku <GPU_SKU>] \
  --email <user_email> \
  --script_output_path <project_root>/Script/<run_name>.sh \
  -- \
  [all stage-specific args for the pipeline script...]
```

Show the generated script to the user for review.

### Step 6: Submit

After user confirms: `sbatch <script_path>`

Report the job ID and monitoring commands:
- `squeue -u $USER` to check job status
- `sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize`
- Logs: `<out_dir>/<run_name>/Inference/logs/` or `Evaluation/logs/` or `Plots/logs/`

### Step 7: Post-Submission Guidance

After inference completes, generate h5mu structure files:
```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 SKILL_DIR/scripts/generate_h5mu_structure.py dummy --adata_dir <out_dir>/<run_name>/Inference/adata
```

Then guide the user through the remaining stages in pipeline order:
1. **Evaluation** (Stage 2a) — run statistical tests
2. **Calibration** (Stage 2b) — U-test, CRT, or Matched Cell DE
3. **K-selection plot** (Stage 3a) — compare K values
4. **Program analysis** (Stage 3b) — per-program plots
5. **Perturbed gene analysis** (Stage 3c) — per-gene plots
6. **Annotation** (Stage 3d) — LLM-driven program annotation
7. **Excel summary** (Stage 3e) — compile results

## Important Notes

- All generated SLURM scripts go into `<project_root>/Script/` — a sibling of `Result/`, NOT inside it.
- Always use `eval "$(conda shell.bash hook)"` before conda activate in any Bash command.
- For torch-cNMF, the generator auto-adds `--gres=gpu:1` and `-C GPU_SKU:<selected_sku>` (default `V100S_PCIE`; use Step 4 to recommend the right GPU).
- The pipeline saves config YAMLs automatically; no need to create them manually.
- Run name convention: `MMDDYY_<short_description>`.
- Output MuData files are at `<out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>.h5mu` with corresponding `_structure.txt` summaries.
- Generated inference scripts automatically run h5mu structure generation after completion. Pass `--no_structure` to skip.
- Known typo in sk-cNMF: `--run_complie_annotation` (not compile). torch-cNMF uses `--run_compile_annotation`.
- Known typo in program analysis: `--top_enrichned_term` (use as-is).
- sk-cNMF `--tol` default is `1e4` (likely a bug; recommend `1e-4`).
- torch-cNMF "online" mode has been renamed to "minibatch". All `--online_*` parameters are now `--minibatch_*`.
