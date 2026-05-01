---
name: run-tests
description: >
  Run the cNMF benchmarking pipeline test suite end-to-end: clean previous outputs,
  run sk-cNMF (CPU) and torch-cNMF (GPU via SLURM) inference tests, evaluation tests,
  and summarize pass/fail results. Use this skill whenever the user says "run tests",
  "run the tests", "test the pipeline", "rerun tests", "run sk-cNMF tests",
  "run torch tests", "run inference tests", "run evaluation tests", or any variation
  asking to execute, rerun, or check the cNMF test suite. Also trigger when the user
  asks to "clean and rerun", "test sk and torch", or "submit test jobs".
---

# Run Tests Skill

Run the cNMF benchmarking pipeline test suite. This cleans previous output, runs tests,
and summarizes results.

## Environment

- **Pipeline root**: `/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF`
- **Inference tests**: `tests/Script/Stage1_Inference/`
- **Evaluation tests**: `tests/Script/Stage2_Evaluation/`
- **Output directory**: `tests/output/`
- **Mini test dataset**: `tests/data/mini_ccperturb.h5ad`
- **Inference conftest**: `tests/Script/Stage1_Inference/conftest.py` (fixtures, shared params)
- **Evaluation conftest**: `tests/Script/Stage2_Evaluation/conftest.py` (h5mu loading, --inference-path)

## Test Suite

### Inference Tests

| Test File | Backend | Runs on | Tests | Command |
|-----------|---------|---------|-------|---------|
| `tests/Script/Stage1_Inference/test_inference_sklearn.py` | sk-cNMF | CPU (direct or SLURM) | 9 (end-to-end + output assertion) | `python -m pytest tests/Script/Stage1_Inference/test_inference_sklearn.py -v --tb=short` |
| `tests/Script/Stage1_Inference/test_inference_parallel_sklearn.py` | sk-cNMF | CPU (direct or SLURM) | 2 (per-K factorize + merge) | `python -m pytest tests/Script/Stage1_Inference/test_inference_parallel_sklearn.py -v --tb=short` |
| `tests/Script/Stage1_Inference/test_inference_torch.py` | torch-cNMF | GPU (SLURM) | 2 per mode (batch/minibatch/dataloader) | `sbatch tests/Script/Stage1_Inference/run_gpu_test_{batch,minibatch,dataloader}.sh` |
| `tests/Script/Stage1_Inference/test_inference_parallel_torch.py` | torch-cNMF | GPU (SLURM) | 2 (per-K factorize + merge) | `sbatch tests/Script/Stage1_Inference/run_gpu_test_parallel.sh` |
| `tests/Script/Stage1_Inference/test_edge_cases.py` | torch-cNMF | GPU (SLURM) | 13 edge cases | `sbatch tests/Script/Stage1_Inference/run_gpu_test_edge_cases.sh` |

### Evaluation Tests

| Test File | Backend | Runs on | Command |
|-----------|---------|---------|---------|
| `tests/Script/Stage2_Evaluation/test_evaluation.py` | NMF_Benchmarking | CPU (direct or SLURM) | `python -m pytest tests/Script/Stage2_Evaluation/test_evaluation.py -v --tb=short` |

### Edge case tests (test_edge_cases.py)

| Test | What it validates |
|------|-------------------|
| `test_minibatch_shuffle` | minibatch mode with shuffle=True |
| `test_minibatch_no_shuffle` | minibatch mode with shuffle=False |
| `test_custom_seeds` | Custom NMF seed array (must match n_iter) |
| `test_cpu_only` | use_gpu=False CPU fallback |
| `test_remove_noncoding` | Pre-filtered ENSG genes |
| `test_minibatch_size_larger_than_data` | minibatch_size >> n_cells |
| `test_numhvgenes_larger_than_total` | numhvgenes >> total genes |
| `test_densify` | densify=True (dense matrix) |
| `test_dataloader_mode_edge` | dataloader mode with small minibatch_size=100 |
| `test_multiple_thresholds` | sel_thresh=[0.4, 0.8, 2.0] all produce output |
| `test_regularization` | Non-zero alpha_usage, alpha_spectra, l1_ratio |
| `test_double_precision` | fp_precision="double" (float64) |
| `test_reproducibility` | Two runs with same seed produce identical spectra |

## SLURM Scripts

All SLURM scripts are in `tests/Script/Stage1_Inference/` and `tests/Script/Stage2_Evaluation/`:

| Script | What it runs | Partition |
|--------|-------------|-----------|
| `Stage1_Inference/run_inference_test_sklearn.sh` | sk-cNMF end-to-end | engreitz,owners (CPU) |
| `Stage1_Inference/run_test_parallel_sklearn.sh` | sk-cNMF parallel | engreitz,owners (CPU) |
| `Stage1_Inference/run_gpu_test_batch.sh` | torch batch mode | gpu,owners (GPU) |
| `Stage1_Inference/run_gpu_test_minibatch.sh` | torch minibatch mode | gpu,owners (GPU) |
| `Stage1_Inference/run_gpu_test_dataloader.sh` | torch dataloader mode | gpu,owners (GPU) |
| `Stage1_Inference/run_gpu_test_parallel.sh` | torch parallel mode | gpu,owners (GPU) |
| `Stage1_Inference/run_gpu_test_edge_cases.sh` | torch edge cases | gpu,owners (GPU) |
| `Stage2_Evaluation/run_eval_test.sh` | evaluation unit tests | engreitz,owners (CPU) |

## Conda Environments

| Environment | Used By |
|-------------|---------|
| `sk-cNMF` | sk-cNMF inference tests |
| `torch-nmf-dl` | torch-cNMF inference tests |
| `NMF_Benchmarking` | evaluation tests |

## Workflow

Follow these steps in order. All commands must run from the pipeline root directory.

### Step 1: Ask the user what to run

Ask the user which tests to run. Options:
- **Inference — sk-cNMF** (CPU) — end-to-end and/or parallel
- **Inference — torch-cNMF** modes: `batch`, `minibatch`, `dataloader`, `parallel`, `edge-cases`
- **Evaluation** — unit tests for evaluation pipeline

If the user already specified (e.g., "run all tests", "run sk and torch batch"), skip asking.
Default if unspecified: run sk-cNMF end-to-end + all torch-cNMF modes.

### Step 2: Clean previous output (MANDATORY)

**ALWAYS remove previous test output before submitting inference jobs.** Tests depend on clean state and will produce incorrect results or fail silently if stale output exists. This step is non-negotiable.

```bash
cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF

# sk-cNMF:
rm -rf tests/output/sk-cNMF
rm -rf tests/output/sk-cNMF-parallel

# torch-cNMF:
rm -rf tests/output/torch-cNMF/batch
rm -rf tests/output/torch-cNMF/minibatch
rm -rf tests/output/torch-cNMF/dataloader
rm -rf tests/output/torch-cNMF-parallel
rm -rf tests/output/torch-cNMF-edge-cases

# Evaluation (only if re-running eval):
rm -rf tests/output/torch-cNMF/dataloader/Evaluation
```

### Step 3: Run sk-cNMF tests (if selected)

Run directly on the current node (CPU only, ~30 seconds each):

```bash
cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF

# End-to-end:
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python -m pytest tests/Script/Stage1_Inference/test_inference_sklearn.py -v --tb=short

# Parallel mode:
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python -m pytest tests/Script/Stage1_Inference/test_inference_parallel_sklearn.py -v --tb=short
```

Or submit via SLURM:

```bash
sbatch tests/Script/Stage1_Inference/run_inference_test_sklearn.sh
sbatch tests/Script/Stage1_Inference/run_test_parallel_sklearn.sh
```

### Step 4: Submit torch-cNMF tests (if selected)

Each test has its own SLURM script:

```bash
sbatch tests/Script/Stage1_Inference/run_gpu_test_batch.sh
sbatch tests/Script/Stage1_Inference/run_gpu_test_minibatch.sh
sbatch tests/Script/Stage1_Inference/run_gpu_test_dataloader.sh
sbatch tests/Script/Stage1_Inference/run_gpu_test_parallel.sh
sbatch tests/Script/Stage1_Inference/run_gpu_test_edge_cases.sh
```

Logs go to `tests/output/torch-cNMF/<mode>/Inference/logs/` (or `torch-cNMF-parallel/logs/`, `torch-cNMF-edge-cases/logs/`).

### Step 5: Run evaluation tests (if selected)

Requires inference output to exist. Auto-detects from: `torch-cNMF/batch/` (preferred), `torch-cNMF/dataloader/`, `torch-cNMF/minibatch/`, `sk-cNMF/`.

```bash
# Direct:
eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking && python -m pytest tests/Script/Stage2_Evaluation/test_evaluation.py -v --tb=short

# Or via SLURM:
sbatch tests/Script/Stage2_Evaluation/run_eval_test.sh
```

### Step 6: Summarize results

Present a summary table with pass/fail, runtime, job IDs, and output locations.

### Step 7: If user wants to wait for SLURM results

Poll `squeue -j <JOB_ID>`, then read the log and verify output files.

## Output structure

```
tests/
├── data/mini_ccperturb.h5ad
├── Script/
│   ├── create_mini_dataset.py
│   ├── Stage1_Inference/
│   │   ├── conftest.py
│   │   ├── test_inference_sklearn.py
│   │   ├── test_inference_parallel_sklearn.py
│   │   ├── test_inference_torch.py
│   │   ├── test_inference_parallel_torch.py
│   │   ├── test_edge_cases.py
│   │   ├── run_inference_test_sklearn.sh
│   │   ├── run_test_parallel_sklearn.sh
│   │   ├── run_gpu_test_batch.sh
│   │   ├── run_gpu_test_minibatch.sh
│   │   ├── run_gpu_test_dataloader.sh
│   │   ├── run_gpu_test_parallel.sh
│   │   └── run_gpu_test_edge_cases.sh
│   └── Stage2_Evaluation/
│       ├── conftest.py
│       ├── test_evaluation.py
│       └── run_eval_test.sh
└── output/
    ├── sk-cNMF/Inference/
    ├── sk-cNMF-parallel/
    ├── torch-cNMF/{batch,minibatch,dataloader}/Inference/
    ├── torch-cNMF-parallel/
    └── torch-cNMF-edge-cases/
```

## Troubleshooting

- **Zero components after density filtering** — increase `n_iter` (need >= 5 for thresh=2.0)
- **ModuleNotFoundError: No module named 'cnmf'** in torch tests — `format_checking.py` should not import `cnmf`
- **sk-cNMF output empty** — run pytest from the pipeline root directory
- **Custom seeds error** — `nmf_seeds` array length must equal `n_iter`
- **Evaluation tests skip** — need inference output from at least one mode (torch-cNMF/batch preferred, then dataloader, minibatch, sk-cNMF)
