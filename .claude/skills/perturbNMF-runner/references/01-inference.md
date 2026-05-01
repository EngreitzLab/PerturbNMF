# Inference Stage (sk-cNMF / torch-cNMF)

## Step A: Get Data Path & Validate

Ask for the input data file path. Then run validation:

```bash
eval "$(conda shell.bash hook)" && conda activate sk-cNMF && python3 SKILL_DIR/scripts/validate_data.py --counts_fn "<path>"
```

Add `--categorical_key`, `--guide_names_key`, etc. if user specifies non-default keys. Report validation results and help resolve issues before proceeding.

## Step B: Collect Parameters

### Tier 1 — Common parameters (always ask, show defaults)

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

### Workflow flags

For a first run, include all:
- `--run_factorize --run_refit --run_compile_annotation --run_diagnostic_plots` (torch-cNMF)
- `--run_factorize --run_refit --run_complie_annotation --run_diagnostic_plots` (sk-cNMF — note typo)

For a rerun after factorization: `--run_refit --run_compile_annotation --run_diagnostic_plots`

### Tier 2 — Advanced parameters

Present a summary and ask if user wants to change any. Defaults are fine for most runs.

#### Both sk-cNMF and torch-cNMF

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss` | `frobenius` | NMF loss function |
| `--nmf_seeds_path` | None | Custom NMF seeds file (text file with one integer per line) |
| `--num_gene` | `300` | Top genes for annotation |
| `--gene_names_key` | None | Column in adata.var with gene names for compiled results (e.g. `symbol`) |

#### torch-cNMF only

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

#### torch-cNMF batch mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_max_epoch` | `500` | Max epochs for batch NMF |
| `--batch_hals_tol` | `0.05` | HALS tolerance |
| `--batch_hals_max_iter` | `200` | Max HALS inner iterations |

#### torch-cNMF minibatch mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--minibatch_max_epoch` | `20` | Max epochs through data |
| `--minibatch_size` | `5000` | Minibatch size |
| `--minibatch_max_iter` | `200` | Max iterations per minibatch |
| `--minibatch_usage_tol` | `0.05` | Usage update tolerance |
| `--minibatch_spectra_tol` | `0.05` | Spectra update tolerance |

## Step C: Estimate Resources & GPU Selection

### GPU Selection (torch-cNMF only)

**Check available GPUs**: `sinfo -p gpu,owners -o "%P %G %f %a" | grep GPU_SKU | sort -u`

**Estimate VRAM**: `(cells x genes x 8 x 3) x 1.5` (data + W + H + 50% overhead)

| Cells | Max K | Recommended GPU |
|-------|-------|-----------------|
| <20k | <=50 | Any (RTX 2080Ti 11GB) |
| 20k-100k | <=200 | RTX 3090 (24GB) or V100S (32GB) |
| 100k-200k | <=300 | V100S (32GB) or A100 (40GB) |
| 200k-500k | <=300 | A100 (40/80GB) or L40S (48GB) |
| >500k | any | H100 (80GB) or use sk-cNMF (CPU) |

Present recommendation with VRAM estimate, confirm with user, pass `--gpu_sku` to generator.

### Memory & Time

**sk-cNMF**: <50k: 64G/4-6h | 50k-200k: 128-256G/8-14h | >200k: 256-512G/14-24h
**torch-cNMF**: <100k: 64G/2-4h | 100k-300k: 128G/6-10h | >300k: 256G/10-15h

### CPUs & Partitions

- sk-cNMF: 1-10 CPUs, partition `engreitz,owners`
- torch-cNMF: 1 CPU (GPU-bound), partition `gpu,owners`
- Memory >256G: add `bigmem`
