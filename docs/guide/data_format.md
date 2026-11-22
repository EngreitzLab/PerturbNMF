# Input data format

What each stage expects to find in its input files. Stage 1 reads an AnnData `.h5ad`; Stages 2 and 3 read the
MuData `.h5mu` that Stage 1 writes, so if Stage 1 ran, their main input is already in the right shape.

The key names below are the defaults. Each one can be changed with a flag (for example `--categorical_key`,
`--guide_names_key`, `--guide_targets_key`, `--guide_assignment_key`); see {doc}`torch_cnmf` and {doc}`metrics`.

## Stage 1 input: AnnData

```
adata
├── X                    raw mRNA counts (cells × genes)
├── obs
│   ├── obs_names        cell IDs
│   └── sample           condition of each cell
├── var
│   └── var_names        gene names
├── uns
│   ├── guide_names      one entry per guide
│   └── guide_targets    gene targeted by each guide
└── obsm
    ├── guide_assignment cells × guides
    ├── X_pca            embedding (for plots)
    └── X_umap           embedding (for plots)
```

| Field | Requirement |
|---|---|
| `X` | Raw, unnormalized integer counts. cNMF does its own normalization (TPM), so do not pass log-normalized or scaled data. Filtering out low-quality cells and genes beforehand is fine. |
| `obs['sample']` | The condition of each cell (timepoint, stimulus, donor, ...). Required even if there is only one condition: fill it with a single value. |
| `var_names` | The gene names you want to see in plots and enrichment results, either Ensembl IDs or gene symbols. |
| `uns['guide_names']` | Name of each guide. Same length as `guide_targets`. |
| `uns['guide_targets']` | The gene each guide targets. Same length and order as `guide_names`, and spelled exactly as in `var_names` so targets can be matched to genes. Control guides share one label, such as `non-targeting`. |
| `obsm['guide_assignment']` | Cells × guides matrix. Its columns must be in the same order as `guide_names`. Watch for this when importing assignments from another tool, for example a SCEPTRE guide assignment matrix. |
| `obsm['X_pca']`, `obsm['X_umap']` | Embeddings used for the UMAP panels in Stage 3. Optional: Stage 3 computes them if they are missing. |

Data without perturbations (no `uns` guide keys or `guide_assignment`) and mouse data also work; see
[Non-perturbed and mouse data](https://github.com/EngreitzLab/PerturbNMF#non-perturbed-and-mouse-data) in the README.

## Stage 1 output: MuData

Stage 1 writes one `cNMF_{K}_{threshold}.h5mu` per K and density threshold, with two modalities:

```
mdata
├── rna                  the input AnnData (X, obs, var, uns, obsm)
└── cNMF
    ├── X                cells × programs (program usage)
    ├── obs, obsm, uns   copied from rna
    ├── var_names        program IDs
    └── varm['loadings'] programs × genes (gene loadings)
```

Stages 2 and 3 read this file directly.

## Stage 2 inputs

Besides the `.h5mu`, each optional test needs its own resource file:

| Test | File | Flag |
|---|---|---|
| Explained variance | TPM-normalized counts written by Stage 1 (`.h5ad`) | `--X_normalized_path` |
| Trait enrichment | GWAS gene–trait table (e.g. OpenTargets L2G) | `--gwas_data_path` |
| Motif enrichment | Enhancer–gene links (e.g. from E2G), genome `.fa`, gene annotation `.gtf`, motif `.meme` | `--enhancer_links`, `--genome_fasta`, `--gene_annotation`, `--motif_file`; see {doc}`metrics` |
| Perturbation association (U-test) | Guide annotation TSV with columns `guide_names`, `guide_targets` and `targeting` (`True` for targeting guides, `False` for controls) | `--guide_annotation_path`, or name the control label with `--guide_annotation_key` |

Each test writes its results to `Evaluation/{K}_{threshold}/`; the full list of files is in
[the metrics guide](metrics.md#output-organization).

## Stage 3 inputs

Plotting and the Excel summary read the Stage 1 `.h5mu` and the Stage 2 `Evaluation/` folder, so there is nothing
new to prepare. See {doc}`plotting` and {doc}`summarization` for what each report contains.
