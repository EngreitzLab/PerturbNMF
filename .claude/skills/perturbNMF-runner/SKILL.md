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
GWAS_DATA=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz
REFERENCE_GTF=/oak/stanford/groups/engreitz/Users/opushkar/genome/IGVFFI9573KOZR.gtf.gz
```

## Default Directory Structure

```
<project_root>/
├── Data/                          # input data (.h5ad files)
├── Result/                        # output directory (--output_directory / --out_dir)
│   └── <run_name>/
│       ├── Inference/             # Stage 1 output
│       └── Evaluation/            # Stage 2 output
└── Script/                        # all generated SLURM .sh scripts
```

Key: `--output_directory`/`--out_dir` -> `Result/`, `--script_output_path` -> `Script/<name>.sh` (sibling of Result/, NOT inside it).

## Step 1: Identify the Stage

Ask which stage to run (or infer from context). Then **read the matching reference file** before collecting parameters.

| Stage | `--stage` value | Conda Env | Reference File |
|-------|-----------------|-----------|----------------|
| sk-cNMF inference | `inference-sk` | `sk-cNMF` | `references/01-inference.md` |
| torch-cNMF inference | `inference-torch` | `torch-cNMF` | `references/01-inference.md` |
| Evaluation | `evaluation` | `NMF_Benchmarking` | `references/02-evaluation.md` |
| U-test calibration | `u-test-calibration` | `NMF_Benchmarking` | `references/03-calibration.md` |
| CRT calibration | `crt-calibration` | `programDE` | `references/03-calibration.md` |
| Matched Cell DE | `matched-cell-de` | `programDE` | `references/03-calibration.md` |
| K-Selection Plot | `k-selection` | `torch-cNMF` | `references/04-visualization.md` |
| Program Analysis Plot | `program-analysis` | `NMF_Benchmarking` | `references/04-visualization.md` |
| Perturbed Gene Plot | `perturbed-gene` | `NMF_Benchmarking` | `references/04-visualization.md` |
| Annotation | `annotation` | `progexplorer` | `references/05-annotation-summary.md` |
| Excel Summary | `excel-summary` | `NMF_Benchmarking` | `references/05-annotation-summary.md` |

**Pipeline flow:** Input (.h5ad) -> Stage 1 (Inference) -> Stage 2a (Evaluation) -> Stage 2b (Calibration) -> Stage 3 (Plots + Annotation + Summary)

## Steps 2-4: Stage-Specific Configuration

Read the reference file listed above for the selected stage. It contains:
- Data validation steps (inference only)
- Parameter tables (required + optional)
- Resource estimation guidelines (memory, CPUs, GPUs, time, partitions)

For edge cases or the full argparse parameter list, read `references/parameter-catalog.md`.
For input/output data format details, read `references/data-format-spec.md`.

## Step 5: Generate SLURM Script

```bash
python3 SKILL_DIR/scripts/generate_slurm.py \
  --stage <stage_value> \
  --job_name <run_name> \
  --output_dir <out_dir> \
  --run_name <run_name> \
  --cpus <N> --mem <MG> --time <HH:MM:SS> --partition <parts> \
  [--gpu] [--gpu_sku <GPU_SKU>] \
  --email <user_email> \
  --script_output_path <project_root>/Script/<run_name>_<stage>.sh \
  -- \
  [all stage-specific args for the pipeline script...]
```

Show the generated script to the user for review.

## Step 6: Clean Previous Output & Submit

**For inference jobs: ALWAYS remove previous output before submitting.** Tests and inference depend on clean state. Stale output causes incorrect results or silent failures.

```bash
# Remove previous inference output for this run:
rm -rf <out_dir>/<run_name>/Inference
```

After cleaning and user confirms: `sbatch <script_path>`

Report the job ID and monitoring commands:
- `squeue -u $USER` to check job status
- `sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize`
- Logs: `<out_dir>/<run_name>/Inference/logs/` or `Evaluation/logs/` or `Plots/logs/`

## Step 7: Post-Submission Guidance

After inference completes, generate h5mu structure files:
```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 SKILL_DIR/scripts/generate_h5mu_structure.py dummy --adata_dir <out_dir>/<run_name>/Inference/adata
```

Then guide the user through remaining stages in pipeline order:
1. **Evaluation** (2a) -> 2. **Calibration** (2b) -> 3. **K-selection** (3a) -> 4. **Program analysis** (3b) -> 5. **Perturbed gene** (3c) -> 6. **Annotation** (3d) -> 7. **Excel summary** (3e)

## Important Notes

- All generated SLURM scripts go into `<project_root>/Script/` — sibling of `Result/`, NOT inside it.
- Always use `eval "$(conda shell.bash hook)"` before conda activate in any Bash command.
- For torch-cNMF, the generator auto-adds `--gres=gpu:1` and `-C GPU_SKU:<selected_sku>`.
- Run name convention: `MMDDYY_<short_description>`.
- Output MuData: `<out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>.h5mu` with `_structure.txt` summaries.
- Generated inference scripts automatically run h5mu structure generation. Pass `--no_structure` to skip.
- Known typos (use as-is): sk-cNMF `--run_complie_annotation`, program analysis `--top_enrichned_term`.
- sk-cNMF `--tol` default is `1e4` (likely a bug; recommend `1e-4`).
- torch-cNMF "online" mode renamed to "minibatch"; all `--online_*` params are now `--minibatch_*`.
