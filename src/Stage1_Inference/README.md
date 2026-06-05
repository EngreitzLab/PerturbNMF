# Stage 1 — Inference

Run consensus NMF (cNMF) on a single-cell counts matrix to extract gene programs. Two interchangeable implementations are provided; both produce the same downstream artefacts (`{run_name}/Inference/adata/cNMF_{K}_{thresh}.h5mu`) and feed identically into Stage 2 (Evaluation).

## Pipeline options

| Pipeline | Solver | Hardware | When to use | Sub-README |
|---|---|---|---|---|
| **sk-cNMF** | scikit-learn coordinate descent (`cd`) or multiplicative update (`mu`) | CPU | Small/medium datasets, no GPU available, or when reproducing legacy results | [sk-cNMF/README.md](sk-cNMF/README.md) |
| **torch-cNMF** | HALS variant (`halsvar`), `mu`, `hals`, or `bpp` | GPU (CUDA) | Large datasets, fast iteration, dense or minibatch modes | [torch-cNMF/README.md](torch-cNMF/README.md) |

sk-cNMF updates one matrix element at a time (can't be parallelized on GPU). torch-cNMF's HALS solver updates a column of elements per step, which maps well to GPU kernels. Results are comparable between the two solvers.

Benchmarking result: https://docs.google.com/presentation/d/1Z25ew7xrnhXD_eQx7e7eg6vtHx_T4uVD/edit?usp=sharing&ouid=103348313942131245812&rtpof=true&sd=true

## Directory layout

```
Stage1_Inference/
├── README.md              ← this file
├── src/                   ← shared library (used by both pipelines)
│   ├── __init__.py
│   ├── run_cNMF.py        ← run_cnmf_consensus, compile_results, filter_noncoding_genes,
│   │                        rename_all_NMF, get_top_indices_fast, annotate_genes_to_excel
│   └── plot_diagnostics.py ← generate_all_plots (elbow, usage heatmap, loading violins)
├── sk-cNMF/
│   ├── README.md          ← per-flag table for the sk-cNMF pipeline
│   ├── environment.yml    ← conda env `sk-cNMF`
│   ├── Slurm_Version/     ← sk-cNMF_batch_inference_pipeline.py, sk-cNMF_batch.sh, sk-cNMF_parallel.sh
│   └── JupterNote_Version/← interactive walkthrough notebook
└── torch-cNMF/
    ├── README.md          ← per-flag table for the torch-cNMF pipeline
    ├── environment.yml    ← conda env `torch-nmf-dl` (GPU)
    ├── Slurm_Version/     ← torch_cnmf_inference_pipeline.py, torch-cNMF_batch.sh,
    │                        torch-cNMF_online.sh, torch-cNMF_parallel.sh
    └── JupterNote_Version/← interactive walkthrough notebook
```

## How to run

Each pipeline ships in two forms:

- **Slurm version** (`Slurm_Version/*.sh`) — what to submit on Sherlock. Three SLURM templates are shipped: `*_batch.sh` (one job, all K values, batch mode), `*_online.sh` (torch only — minibatch mode), and `*_parallel.sh` (one job per K via SLURM `--array`).
- **Jupyter version** (`JupterNote_Version/*.ipynb`) — interactive walkthrough with the same 13 numbered steps (parameters → load → optional non-coding filter → prepare → factorize → combine → K selection → consensus → MuData → annotation → diagnostic plots → merge parallel K). Useful for debugging or running on small data.

For details on individual CLI flags, see the per-pipeline READMEs linked above.

### Single-job vs parallel mode

For broad K sweeps it is often faster to factorize each K in its own SLURM job, then merge the spectra files before refit + consensus.

| Mode | When to use | How |
|---|---|---|
| Single job | Few K values, fits in one job's time budget | `*_batch.sh` (or `*_online.sh` for torch minibatch) |
| Parallel K array | Many K values, want to run each on its own GPU/CPU | `*_parallel.sh` (array job, one K per task) → then **a separate combiner invocation** with `--parallel_running` |

**Parallel layout (required by `--parallel_running`):**

```
{OUT_DIR}/
└── {RUN_NAME}/
    ├── {RUN_NAME}_30/Inference/cnmf_tmp/   ← per-K parallel outputs
    ├── {RUN_NAME}_50/Inference/cnmf_tmp/
    ├── …
    └── {RUN_NAME}_all/Inference/cnmf_tmp/  ← merged destination (created by --parallel_running)
```

The shipped `*_parallel.sh` invokes the pipeline with `--output_directory "$OUT_DIR/$RUN_NAME"` and `--run_name "${RUN_NAME}_${K}"`, which produces exactly this layout. After all array tasks finish, run the pipeline once more with `--parallel_running --output_directory $OUT_DIR --run_name $RUN_NAME` (no `_${K}` suffix); it calls `rename_all_NMF` to consolidate spectra files into `{RUN_NAME}_all/Inference/cnmf_tmp/`. You can then re-invoke with `--run_name "${RUN_NAME}_all" --run_refit --run_compile_annotation` to finish on the merged directory.

**Do not pass `--parallel_running` for a normal single-job run** — it expects the nested layout and silently merges zero files otherwise.

## Output structure

After a successful run (regardless of which pipeline), the on-disk layout is:

```
{OUT_DIR}/{RUN_NAME}/
└── Inference/
    ├── adata/cNMF_{K}_{thresh}.h5mu      ← primary output (rna + cNMF modalities), feeds Stage 2
    ├── cnmf_tmp/                          ← internal cNMF working files (per-replicate H/W, k_selection_stats, ...)
    ├── Inference.spectra.k_{K}.dt_{thresh}.consensus.txt   ← (sk-cNMF) consensus H
    ├── Inference.gene_spectra_score.k_{K}.dt_{thresh}.txt  ← (torch-cNMF) z-scored gene spectra
    ├── Inference.usages.k_{K}.dt_{thresh}.consensus.txt
    ├── loading/                           ← per-K sparse loading matrices
    ├── prog_data/                         ← per-K program-level summaries
    ├── diagnosis_plots/                   ← elbow curves, usage heatmaps, loading violins (if run_diagnostic_plots)
    ├── Annotation/{K}_{thresh}.xlsx       ← top-N gene annotations per program
    └── config_<SLURM_JOB_ID>.yml          ← snapshot of args + SLURM env for this run
```

For exact column semantics of `.h5mu`, see Stage 2 docs and the per-pipeline READMEs.

## Recommended workflow for K selection

1. Select as many genes as possible as highly variable genes (HVGs), within time and memory constraints, to maximize enriched-term coverage.
2. For larger datasets, use a tighter convergence tolerance and more iterations to maximize solution stability.
3. Run a broad sweep across a wide range of K values to inspect overall K-selection trends in stability and biological metrics.
4. If needed, run targeted sweeps with denser sampling around the best-performing K range identified in step 3.
5. Examine perturbation calibration plots (Stage 2b) to choose the right p-value estimation method and the minimal covariate set.
6. Select the optimal K by integrating stability, biological-term metrics, and gene annotations.
7. Re-test density thresholds (`--sel_threshs`) on the selected K.
8. Generate the full set of downstream analyses and visualizations for the chosen K.

