# Visualization Stages (K-Selection / Program Analysis / Perturbed Gene)

---

## K-Selection Plot (Stage 3a)

**Conda**: `torch-cNMF`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--output_directory` | Directory with cNMF output |
| `--run_name` | cNMF run name |
| `--save_folder_name` | Where to save plots |
| `--eval_folder_name` | Path to Eval results |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--K` | (from data) | K values |
| `--sel_threshs` | (from data) | Density thresholds |
| `--groupby` | `sample` | Grouping variable |
| `--pval` | `0.05` | P-value threshold |
| `--selected_k` | None | K value to highlight |

Enrichment file patterns use `{k}` placeholder (and `{sample}` for perturbation). See `references/parameter-catalog.md` Section 4 for all optional params.

### SLURM resources

- Partition: `engreitz,owners`, CPUs: 4, Memory: 32-64G, Time: 1h

---

## Program Analysis Plot (Stage 3b)

**Conda**: `NMF_Benchmarking`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--mdata_path` | Path to .h5mu file |
| `--perturb_path_base` | Base path for perturbation results |
| `--GO_path` | Path to GO enrichment results |
| `--pdf_save_path` | Output directory for plots |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_program` | `10` | Top programs to display |
| `--top_enrichned_term` | `10` | Top GO terms per program (**note typo**: use as-is) |
| `--p_value` | `0.05` | Significance threshold |
| `--programs` | None | Specific program numbers (e.g. `4 5 6`); if omitted, all plotted |
| `--subsample_frac` | None | Fraction of cells to subsample for UMAP |
| `--PDF` | off | Save as PDF (else SVG) |

See `references/parameter-catalog.md` Section 5 for all optional params.

### SLURM resources

- Partition: `engreitz,owners`, CPUs: 4-8, Memory: 64-128G, Time: 2-4h

---

## Perturbed Gene Plot (Stage 3c)

**Conda**: `NMF_Benchmarking`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--mdata_path` | Path to .h5mu file |
| `--perturb_path_base` | Base path for perturbation results |
| `--save_path` | Output directory for plots |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_n_programs` | `10` | Top programs to display per gene |
| `--top_corr_genes` | `5` | Top correlated genes per program |
| `--significance_threshold` | `0.05` | P-value threshold |
| `--gene_list_file` | None | File with gene names to process (one per line) |
| `--subsample_frac` | None | Fraction of cells to subsample for UMAP |
| `--parallel` | off | Use fork-based multiprocessing (Linux only) |
| `--PDF` | off | Save as PDF (else SVG) |

See `references/parameter-catalog.md` Section 6 for all optional params.

### SLURM resources

- Partition: `engreitz,owners`, CPUs: 4-10, Memory: 64-128G, Time: 2-6h
