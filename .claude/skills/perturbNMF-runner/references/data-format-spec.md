# Input Data Format Specification

## Supported File Formats

- **`.h5ad`** — AnnData (single modality, typically for inference input)
- **`.h5mu`** — MuData (multi-modal, typically post-inference with RNA + cNMF modalities)

## AnnData Structure (Pre-Inference Input)

The input `.h5ad` file for inference should contain:

### X (Expression Matrix)
- Raw or normalized gene expression counts
- Shape: `(n_cells, n_genes)`
- Can be sparse (CSR/CSC) or dense — pipeline handles both

### obs (Cell Metadata)
Required columns:
| Key | Description | Example Values |
|-----|-------------|----------------|
| `sample` or `batch` | Categorical variable for cell grouping (controlled by `--categorical_key`) | `D0`, `sample_D1`, `sample_D2` |

### uns (Unstructured Metadata)
| Key | Type | Description |
|-----|------|-------------|
| `guide_names` | array/list | Names of all guides (length = number of guides) |
| `guide_targets` | array/list | Target gene for each guide (same length as guide_names) |

### obsm (Multi-dimensional Observations)
| Key | Shape | Description |
|-----|-------|-------------|
| `X_pca` | `(n_cells, n_components)` | PCA embeddings |
| `X_umap` | `(n_cells, 2)` | UMAP embeddings |
| `guide_assignment` | `(n_cells, n_guides)` | Binary guide assignment matrix (can be sparse) |

### Consistency Requirements
- `len(uns['guide_names'])` must equal `obsm['guide_assignment'].shape[1]`
- `len(uns['guide_targets'])` must equal `len(uns['guide_names'])`
- Gene names in `var_names` should be gene symbols (not Ensembl IDs) for downstream annotation

## MuData Structure (Post-Inference)

After inference, the pipeline produces `.h5mu` files with two modalities:

### `mdata['rna']` — RNA Expression Modality
Same structure as the AnnData input above.

### `mdata['cNMF']` — Program Modality
| Element | Description |
|---------|-------------|
| `X` | Usage matrix: `(n_cells, K)` — each cell's program usage scores |
| `var` | Program names: one row per program (K rows) |
| `varm['loadings']` | Gene loading matrix: `(K, n_genes)` — how much each gene contributes to each program |
| `obsm['X_pca']` | PCA on programs (copied from RNA) |
| `obsm['X_umap']` | UMAP on programs (copied from RNA) |
| `obsm['guide_assignment']` | Guide assignment (same as RNA) |
| `uns['guide_names']` | Guide names (same as RNA) |
| `uns['guide_targets']` | Guide targets (same as RNA) |

## Guide Annotation File

Optional TSV file with guide metadata. Used for evaluation and calibration.

| Column | Type | Description |
|--------|------|-------------|
| (index) | str | Guide identifier |
| `guide_names` | str | Guide name (must match `uns['guide_names']`) |
| `guide_targets` | str | Target gene name |
| `targeting` | bool | `True` for targeting guides, `False` for non-targeting/safe-targeting controls |

## Reference Files

### Reference GTF
- Standard GTF format (e.g., GENCODE v43)
- Used for validating gene names against genome annotation
- Path: `/oak/stanford/groups/engreitz/Users/opushkar/genome/IGVFFI9573KOZR.gtf.gz`

### GWAS Data
- OpenTargets L2G filtered data
- Used for trait enrichment analysis in evaluation
- Path: `/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz`

### Motif Files
- HOCOMOCO motif database in MEME format
- Used for motif enrichment analysis
- Path: `/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/hocomoco_meme.meme`

### Genome Sequence
- hg38 reference genome FASTA
- Required for motif analysis
- Path: `/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/hg38.fa`
