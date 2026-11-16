# Get started

This walkthrough runs Stage 1 on PBMC3k, the 3k PBMC dataset from scanpy, and takes a few minutes on one GPU.
PBMC3k has no perturbations, so it covers only inference and K selection. For the full
perturbation workflow, see {doc}`tutorials/index`.

## 1. Get the data

```python
import scanpy as sc

adata = sc.datasets.pbmc3k()
adata.write("examples/example_output/pbmc3k.h5ad")
```

## 2. Run torch-cNMF

```bash
eval "$(conda shell.bash hook)" && conda activate torch-nmf-dl

python3 src/Stage1_Inference/torch-cNMF/Slurm_Version/torch_cnmf_inference_pipeline.py \
    --counts_fn examples/example_output/pbmc3k.h5ad \
    --output_directory examples/example_output \
    --run_name pbmc_test \
    --species human \
    --K 5 7 10 \
    --numiter 10 \
    --numhvgenes 2000 \
    --sel_thresh 2.0 \
    --seed 14 \
    --algo halsvar \
    --mode batch \
    --run_factorize --run_refit --run_compile_annotation \
    --run_gene_annotation --run_diagnostic_plots
```

To submit the same command as a SLURM job, use `examples/Inference/pbmc_test.sh`.
For a machine without a GPU, use sk-cNMF instead (see {doc}`guide/sk_cnmf`).

## 3. Look at the output

```
examples/example_output/pbmc_test/Inference/
├── adata/cNMF_{K}_2_0.h5mu     # cell scores + gene loadings, one file per K
├── loading/                    # top genes per program
└── ...                         # K-selection and density-filtering plots
```

Next: {doc}`tutorials/index` · {doc}`guide/inference`
