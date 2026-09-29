---
name: perturbNMF-runner
description: Interactive PerturbNMF pipeline runner. Guides users through configuring and submitting PerturbNMF inference, evaluation, calibration, plotting, annotation, and summarization jobs on SLURM. Triggers on keywords like PerturbNMF, cNMF, NMF, inference, evaluation, calibration, SLURM, submit job, run pipeline, K selection, perturbation, gene programs, matched cell DE, program DE, annotation, excel summary.
user_invocable: true
---

# PerturbNMF Pipeline Runner

You are an interactive assistant for running the PerturbNMF pipeline on a SLURM cluster. Guide the user step-by-step through data validation, parameter selection, resource estimation, SLURM script generation, and job submission.

## Constants

These are site-specific. Resolve each one at the start of the session — from the environment, or by asking the user — and never assume a default path, partition, or email.

**PIPELINE_ROOT: always ask the user**, every session, before generating any script — even if `$PIPELINE_ROOT` is set or the skill's own location looks like the repo. If `$PIPELINE_ROOT` is set, offer it as a suggestion, but have the user confirm it. Pass the confirmed value to `generate_slurm.py` as `--pipeline_root`; the flag is required. The generated script exports it as `PIPELINE_ROOT`.

```
PIPELINE_ROOT=<ask the user>        # PerturbNMF checkout on the machine that runs the job (required, always confirmed)
SKILL_DIR=${PIPELINE_ROOT}/.claude/skills/perturbNMF-runner
EMAIL=${SLURM_MAIL_USER}            # optional; ask the user. If unset, no mail directives are written
PARTITION=${SLURM_PARTITION}        # ask the user which partition(s) to use on their cluster
GWAS_DATA=${PIPELINE_ROOT}/src/Stage2_Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz   # fetched by setup_resources.sh
REFERENCE_GTF=<ask the user>        # optional GENCODE/Ensembl GTF (.gtf.gz) for gene ID/name mapping
```

Conda: always activate with `eval "$(conda shell.bash hook)" && conda activate <env>` — never hardcode a conda install path.

## Default Directory Structure

```
<project_root>/
├── Data/                          # input data (.h5ad files)
├── Result/                        # output directory (--output_directory / --out_dir)
│   └── <run_name>/
│       ├── Inference/             # Stage 1 output
│       ├── Evaluation/            # Stage 2 output
│       └── Interpretation/        # Stage 3 output (plots, annotation, summary)
│           ├── K_selection/       # k-selection plots
│           ├── Program_analysis/  # program analysis plots
│           ├── Perturbed_gene/    # perturbed gene plots
│           ├── Annotation/        # LLM annotation
│           └── Excel_summary/     # excel summary
└── Script/                        # all generated SLURM .sh scripts
```

Key: `--output_directory`/`--out_dir` -> `Result/`, `--script_output_path` -> `Script/<name>.sh` (sibling of Result/, NOT inside it).

For plotting stages, default `--save_folder_name` to `<out_dir>/<run_name>/Interpretation/<Stage_Subdir>/` (e.g. `Interpretation/K_selection/`, `Interpretation/Program_analysis/`). Logs go in the `logs/` subfolder of that same directory.

## Step 1: Identify the Stage

Ask which stage to run (or infer from context). Then **read the matching reference file** before collecting parameters.

| Stage | `--stage` value | Conda Env | Reference File |
|-------|-----------------|-----------|----------------|
| sk-cNMF inference | `inference-sk` | `sk-cNMF` | `references/01-inference.md` |
| torch-cNMF inference | `inference-torch` | `torch-nmf-dl` | `references/01-inference.md` |
| Evaluation | `evaluation` | `NMF_Benchmarking` | `references/02-evaluation.md` |
| U-test calibration | `u-test-calibration` | `NMF_Benchmarking` | `references/03-calibration.md` |
| CRT calibration | `crt-calibration` | `NMF_Benchmarking` | `references/03-calibration.md` |
| Matched Cell DE | `matched-cell-de` | `programDE` | `references/03-calibration.md` |
| K-Selection Plot | `k-selection` | `torch-nmf-dl` | `references/04-visualization.md` |
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
  --cpus <N> --mem <MG> --time <HH:MM:SS> [--partition <PARTITION>] \
  [--gpu] [--gpu_min_mem <GB> | --gpu_sku <GPU_SKU>] \
  --pipeline_root <PIPELINE_ROOT> [--email <EMAIL>] \
  --script_output_path <project_root>/Script/<run_name>_<stage>.sh \
  -- \
  [active stage-specific args...] \
  ---COMMENTED--- \
  [all remaining stage flags as `--flag <default_or_example>` ...]
```

**ALWAYS include every flag for the stage in the generated script — this applies to every stage (inference-sk, inference-torch, evaluation, u-test-calibration, crt-calibration, matched-cell-de, k-selection, program-analysis, perturbed-gene, annotation, excel-summary).** Active flags (the ones the user is using) go before the `---COMMENTED---` sentinel and appear inside the python command. All other flags listed for this stage in `references/parameter-catalog.md` go after the sentinel and appear as `#     --flag value` lines below the command, so the user can uncomment to toggle them on later.

- Boolean flags: emit just `--flag_name` (no value) after the sentinel.
- Value flags: emit `--flag_name <default_or_sensible_example>` so the user only has to uncomment + tweak.
- Skip flags that have already been provided as active flags — don't duplicate them in the commented section.
- Use `parameter-catalog.md` (Sections 1–12, by stage) as the authoritative list. If an optional flag exists there but isn't being used, it belongs in the commented section.
- This convention is mandatory — never emit a script that omits available stage flags. The commented block is part of the script's value to the user (toggling features without re-invoking the skill).

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
- For torch-cNMF GPU selection: by default no GPU constraint is written. On clusters that expose `GPU_MEM:<N>GB` node features, pass `--gpu_min_mem <GB>` to select all GPUs with sufficient memory (e.g., `-C "GPU_MEM:32GB|GPU_MEM:40GB|..."`); on clusters with `GPU_SKU:<name>` features, `--gpu_sku` targets one SKU. Ask the user which convention their cluster uses (`sinfo -o "%P %G %f"`), and estimate the minimum VRAM needed.
- Run name convention: `MMDDYY_<short_description>`.
- Output MuData: `<out_dir>/<run_name>/Inference/adata/cNMF_<K>_<sel_thresh>.h5mu` with `_structure.txt` summaries.
- Generated inference scripts automatically run h5mu structure generation. Pass `--no_structure` to skip.
- Known typos (use as-is): sk-cNMF `--run_complie_annotation`, program analysis `--top_enrichned_term`.
- sk-cNMF `--tol` default is `1e4` (likely a bug; recommend `1e-4`).
- torch-cNMF "online" mode renamed to "minibatch"; all `--online_*` params are now `--minibatch_*`.
