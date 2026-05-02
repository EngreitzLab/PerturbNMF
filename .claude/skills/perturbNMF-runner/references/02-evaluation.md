# Evaluation Stage

## Step A: Read inference config to auto-populate parameters

```bash
cat <out_dir>/<run_name>/Inference/config_*.yml
```

Extract: `K`, `sel_thresh`, `categorical_key`, `gene_names_key`, `species` -> `organism`, `data_key`, `prog_key`.

## Step B: Construct paths

- `--out_dir`: Parent directory containing the run (e.g., `Result/`)
- `--run_name`: The run directory name
- `--X_normalized_path`: `<out_dir>/<run_name>/Inference/cnmf_tmp/Inference.norm_counts.h5ad`

## Step C: Determine which tests to run (9 metrics total)

| Flag | Description | Notes |
|------|-------------|-------|
| `--Perform_categorical` | Categorical association (Kruskal-Wallis + Dunn's) | |
| `--Perform_perturbation` | Perturbation sensitivity | **Requires guide data; skip for bulk RNA-seq** |
| `--Perform_motif` | TF motif enrichment | |
| `--Perform_trait` | GWAS trait enrichment | Requires `--gwas_data_path` |
| `--Perform_geneset` | GO + geneset enrichment (Reactome, MsigDB) | |
| `--Perform_explained_variance` | Explained variance per K | Needs `--X_normalized_path` |

Reconstruction error and stability are computed automatically.

## Step D: Optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gwas_data_path` | None | Path to GWAS data (use GWAS_DATA constant) |
| `--gene_names_key` | `symbol` | Column in data_guide["rna"].var with gene names |
| `--FDR_method` | `StoreyQ` | FDR correction: `StoreyQ` or `BH` |
| `--organism` | `human` | Species for enrichment |
| `--n_top` | `300` | Top genes for enrichment tests |
| `--guide_annotation_key` | `["non-targeting"]` | Non-targeting guide identifiers (accepts multiple values) |

## SLURM Resources

- Partition: `engreitz,owners,bigmem`
- CPUs: 10-20
- Memory: 64-256G
- Time: 3-5h
