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
| K | list of int | [30, 50, 60, 80, 100, 200, 250, 300] | Values of K (number of components) to run NMF for |
| numiter | int | 10 | Number of NMF replicates to run |
| seed | int | 14 | Random seed for reproducibility |
| loss | str | "frobenius" | Loss function: "frobenius" (L2), "kullback-leibler" (KL), "itakura-saito" (IS), or float |
| numhvgenes | int | 5451 | Number of highly variable genes to use for factorization |
| algo | str | "mu" | Algorithm: "mu" (multiplicative update) or "cd" (coordinate descent) |
| init | str | "random" | Initialization method: "random", "nndsvd", "nndsvda", "nndsvdar" |
| max_NMF_iter | int | 500 | Maximum number of iterations per individual NMF run |
| tol | float | 1e4 | Tolerance for NMF convergence |
| sel_thresh | list of float | [2.0] (fallback: [0.4, 0.8, 2.0]) | Density threshold(s) for consensus selection |
| nmf_seeds_path | str | None | Path to .npy file containing custom NMF seeds for reproducibility |

## Annotation and Compilation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| species | str | - | Species for gene annotation (required) |
| check_format | flag | False | Validate input data format before running |
| parallel_running | flag | False | Compile files after parallel mode for multiple K values |
| num_gene | int | 300 | Number of top genes to use for program annotation |
| run_refit | flag | False | Run the combine and consensus steps after factorization |
| run_complie_annotation | flag | False | Compile results and generate gene annotations for all K values |
| run_factorize | flag | False | Run the NMF factorization step |
| run_diagnostic_plots | flag | False | Generate diagnostic plots (elbow curves, usage heatmaps, loading violins) after inference |
| guide_annotation_path | str | None | Path to file with guide annotations (optional) |
| reference_gtf_path | str | None | Path to reference GTF file for gene name validation (optional) |

## Data Access Keys

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_key | str | "rna" | Key to access gene expression data in MuData object |
| prog_key | str | "cNMF" | Key to access cNMF programs in MuData object |
| categorical_key | str | "sample" | Key to access cell condition information in obs |
| guide_names_key | str | "guide_names" | Key to access guide names in uns |
| guide_targets_key | str | "guide_targets" | Key to access guide targets in uns |
| guide_assignment_key | str | "guide_assignment_key" | Key to access guide assignments in obsm |
