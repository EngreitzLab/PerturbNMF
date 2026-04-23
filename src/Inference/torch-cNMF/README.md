# torch-cNMF

* GPU powered cNMF
* Individual NMF inference using: [NMF-Torch](https://github.com/lilab-bcb/nmf-torch)
* consensus NMF using: [torch-cNMF](https://github.com/ymo6/torch_based_cNMF)
* To run torch-cNMF, create a new conda environment with `conda env create -f environment.yml` with the provided yml file, then run `pip install git+https://github.com/ymo6/torch_based_cNMF.git` and `pip install git+https://github.com/ymo6/nmf-torch.git` in the terminal
* Singularity Container can be found in https://hub.docker.com/r/igvf/torch-cnmf/tags

## Required I/O Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| counts_fn | str | Path to input counts matrix (.h5ad, .mtx, .mtx.gz, .npz, or tab-delimited text) |
| output_directory | str | Path to output directory where results will be saved |
| run_name | str | Name for this cNMF run (used for output file naming) |

## cNMF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| K | list of int | [5, 7, 10] | Values of K (number of components) to run NMF for |
| numiter | int | 10 | Number of NMF iterations per K value |
| densify | flag | False | Densify sparse matrix before factorization |
| tpm_fn | str | None | Path to TPM normalized data (optional) |
| seed | int | 14 | Random seed for reproducibility |
| loss | str | "frobenius" | Loss function: "frobenius", "kullback-leibler", or "itakura-saito" |
| numhvgenes | int | 2000 | Number of highly variable genes to use for factorization |
| genes_file | str | None | Path to file containing list of genes to use instead of HVG selection |
| use_gpu | flag | False | Use GPU acceleration if available |
| algo | str | "halsvar" | Algorithm: "mu" (multiplicative update), "hals", "halsvar" (HALS variant mimicking BPP), "bpp" (Block Principal Pivoting) |
| mode | str | "batch" | Learning mode: "batch", "minibatch", or "dataloader". minibatch/dataloader only work with frobenius loss |
| init | str | "random" | Initialization method: "random" or "nndsvd" |
| tol | float | 1e-4 | Tolerance for convergence check |
| n_jobs | int | -1 | Number of CPU threads for PyTorch (-1 uses PyTorch default) |
| fp_precision | str | "float" | Numeric precision: "float" (32-bit) or "double" (64-bit) |
| nmf_seeds_path | str | None | Path to .npy file containing custom NMF seeds for reproducibility |
| sel_thresh | list of float | [2.0] | Density threshold(s) for consensus selection |

## Regularization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| alpha_usage | float | 0.0 | Regularization strength on usage matrix W |
| alpha_spectra | float | 0.0 | Regularization strength on spectra matrix H |
| l1_ratio_usage | float | 0.0 | L1 penalty ratio on W (0=L2 only, 1=L1 only) |
| l1_ratio_spectra | float | 0.0 | L1 penalty ratio on H (0=L2 only, 1=L1 only) |

## Batch Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_max_epoch | int | 500 | Maximum epochs for batch learning |
| batch_hals_tol | float | 0.05 | HALS tolerance for halsvar algorithm |
| batch_hals_max_iter | int | 200 | Maximum HALS iterations per H/W update |

## Minibatch Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| minibatch_max_epoch | int | 20 | Maximum passes over all data |
| minibatch_size | int | 5000 | Size of mini-batches |
| minibatch_max_iter | int | 200 | Maximum iterations for H/W update per mini-batch |
| minibatch_usage_tol | float | 0.05 | Convergence tolerance for usage updates |
| minibatch_spectra_tol | float | 0.05 | Convergence tolerance for spectra updates |
| minibatch_shuffle | flag | False | Enable shuffling of samples across mini-batches each epoch |

## Refit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| sk_cd_refit | flag | False | Use scikit-learn coordinate descent for refitting |

## Preprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| remove_noncoding | flag | False | Remove non-coding genes whose symbol starts with an Ensembl ID prefix |
| ensembl_prefix | str | "ENSG" | Ensembl ID prefix used to identify non-coding genes |
| gene_symbol_key | str | "symbol" | Column in adata.var containing gene symbols |

## Annotation and Compilation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| species | str | - | Species for gene annotation (required) |
| parallel_running | flag | False | Compile files after parallel mode for multiple K values |
| num_gene | int | 300 | Number of top genes to include in annotation |
| run_factorize | flag | False | Run the NMF factorization step |
| run_refit | flag | False | Run the refit step (combine, k_selection_plot, and consensus) |
| run_compile_annotation | flag | False | Run the compilation and annotation step |
| run_diagnostic_plots | flag | False | Generate diagnostic plots (elbow curves, usage heatmaps, loading violins) after inference |
| gene_names_key | str | None | Column in adata.var with gene names (e.g. "symbol"). If set, compiled results store these as var_names instead of default var index |

## Data Access Keys

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_key | str | "rna" | Key to access gene expression data in MuData object |
| prog_key | str | "cNMF" | Key to access cNMF programs in MuData object |
| categorical_key | str | "sample" | Key to access cell condition information in obs |
| guide_names_key | str | "guide_names" | Key to access guide names in uns |
| guide_targets_key | str | "guide_targets" | Key to access guide targets in uns |
| guide_assignment_key | str | "guide_assignment" | Key to access guide assignments in obsm |
