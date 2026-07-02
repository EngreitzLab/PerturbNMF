# sk-cNMF

* Individual NMF inference using: [sklearn.decomposition.non_negative_factorization](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.non_negative_factorization.html)
* consensus NMF using [sk-cNMF](https://github.com/EngreitzLab/sk_cNMF) which is a slightly modified version from the [Orginal cNMF](https://github.com/dylkot/cNMF/tree/main) with more flexiblity to choose solver and loss function. 
* To run sk-cNMF, create a new conda environment with `conda env create -f environment.yml --name sk-cNMF` with the provided yml file, then run `pip install git+https://github.com/EngreitzLab/sk_cNMF.git` in the terminal

## Required I/O Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| counts_fn | str | Path to input counts matrix (.h5ad, .h5mu, .mtx, .mtx.gz, .npz, or tab-delimited text) |
| output_directory | str | Directory where all outputs will be saved |
| run_name | str | Name for this cNMF run (used for output file naming) |

## cNMF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| K | list of int | [30, 50, 70, 80, 100, 200, 300] | Values of K (number of components) to run NMF for |
| numiter | int | 10 | Number of NMF replicates to run |
| seed | int | 14 | Random seed for reproducibility |
| loss | str | "frobenius" | Loss function: "frobenius" (L2), "kullback-leibler" (KL), "itakura-saito" (IS), or float |
| numhvgenes | int | 5451 | Number of highly variable genes to use for factorization |
| algo | str | "mu" | Algorithm: "mu" (multiplicative update) or "cd" (coordinate descent) |
| init | str | "random" | Initialization method: "random", "nndsvd", "nndsvda", "nndsvdar" |
| max_NMF_iter | int | 500 | Maximum number of iterations per individual NMF run |
| tol | float | 1e4 | Tolerance for NMF convergence |
| sel_threshs | list of float | [0.2, 2.0] | Density threshold(s) for consensus selection |
| nmf_seeds_path | str | None | Path to .npy file containing custom NMF seeds for reproducibility |

## Annotation and Compilation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| species | str | - | Species for gene annotation (required) |
| parallel_running | flag | False | Combine per-K spectra files from a parallel run into `{RUN_NAME}_all/Inference/cnmf_tmp/`. **Only use when the parallel jobs were submitted via `Slurm_Version/sk-cNMF_parallel.sh`** — it expects the nested layout `{OUT_DIR}/{RUN_NAME}/{RUN_NAME}_{K}/Inference/cnmf_tmp/`. Do not pass this flag for normal single-job runs; the merge will not find any files. |
| num_gene | int | 300 | Number of top genes to use for program annotation |
| run_refit | flag | False | Run the combine and consensus steps after factorization |
| run_complie_annotation | flag | False | Compile results and generate gene annotations for all K values |
| run_factorize | flag | False | Run the NMF factorization step |
| run_diagnostic_plots | flag | False | Generate diagnostic plots (elbow curves, usage heatmaps, loading violins) after inference |
| skip_existing | flag | False | If set, skip NMF replicates already completed on disk (pause/resume mode). Default re-runs all replicates from scratch |

## Preprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| remove_noncoding | flag | False | Remove non-coding genes before factorization. With `gtf_path`, keeps only genes annotated as `protein_coding` in the GTF (matched by Ensembl ID); otherwise falls back to the Ensembl-prefix heuristic on `gene_names_key` |
| ensembl_prefix | str | "ENSG" | Ensembl ID prefix used by the (fallback) prefix-based non-coding filter |
| gtf_path | str | None | Path to a GENCODE/Ensembl GTF(.gz). Enables GTF-based `remove_noncoding` and `add_gene_names_from_gtf` |
| gene_id_key | str | "gene_id" | Column in adata.var holding Ensembl gene IDs (used for GTF-based filtering / gene-name annotation). Falls back to var_names if absent |
| add_gene_names_from_gtf | flag | False | Populate `adata.var[gene_names_key]` with gene symbols looked up from `gtf_path` by Ensembl ID (from `gene_id_key`). Unmatched IDs keep their Ensembl ID. Requires `gtf_path` |

## Data Access Keys

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_key | str | "rna" | Key to access gene expression data in MuData object |
| prog_key | str | "cNMF" | Key to access cNMF programs in MuData object |
| categorical_key | str | "sample" | Key to access cell condition information in obs |
| guide_names_key | str | "guide_names" | Key to access guide names in uns |
| guide_targets_key | str | "guide_targets" | Key to access guide targets in uns |
| guide_assignment_key | str | "guide_assignment_key" | Key to access guide assignments in obsm |
| gene_names_key | str | None | Column in adata.var with gene names to use in compiled results (e.g. 'symbol'). If None, uses var_names |
