# Installation

PerturbNMF is a set of scripts and notebooks, not a pip package. Clone the repository
and create the conda environment for each stage you plan to run.

```bash
git clone https://github.com/EngreitzLab/PerturbNMF.git
cd PerturbNMF
export PIPELINE_ROOT=$PWD
```

## Conda environments

Each stage ships its own `environment.yml`. The stages have conflicting dependencies, so
do not reuse one environment for another.

| Stage | Environment file |
|---|---|
| 1 · sk-cNMF inference (CPU) | `src/Stage1_Inference/sk-cNMF/environment.yml` |
| 1 · torch-cNMF inference (GPU) | `src/Stage1_Inference/torch-cNMF/environment.yml` |
| 2a · Metrics | `src/Stage2_Evaluation/A_Metrics/environment.yml` |
| 2b · U-test calibration | `src/Stage2_Evaluation/B_Calibration/Slurm_version/U-test_perturbation_calibration/environment.yml` |
| 2b · CRT calibration | `src/Stage2_Evaluation/B_Calibration/Slurm_version/CRT/environment.yml` |
| 2b · Matched-cell DE (R) | `src/Stage2_Evaluation/B_Calibration/Slurm_version/Matched_cell_programDE/environment.yml` |
| 3 · Interpretation | `src/Stage3_Interpretation/environment.yml` |

```bash
conda env create -f src/Stage1_Inference/torch-cNMF/environment.yml
eval "$(conda shell.bash hook)" && conda activate <env_name>
```

`<env_name>` is the `name:` field at the top of each file.

## Reference data (Stage 2)

Evaluation needs about 3 GB of reference files that are not stored in git:

| File | Used for |
|---|---|
| `OpenTargets_L2G_Filtered.csv.gz` | Trait (GWAS) enrichment |
| `hocomoco_meme.meme` | Motif enrichment |
| `hg38.fa` | Motif enrichment |

Put them in one directory and link it into the repo:

```bash
RESOURCES_SRC=/path/to/Resources ./setup_resources.sh
```

A reference GTF (for example from GENCODE) is optional. Pass it with `--gtf_path` or `--reference_gtf_path`.

## Running on a cluster

Every stage has a `Slurm_Version/*.sh` runner. Before submitting, fill in the `<partition>` and
`<your_email>` placeholders and export `PIPELINE_ROOT`. See {doc}`guide/hpc`.
