#!/bin/bash
#SBATCH --job-name=extract_prog_info
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/extract_prog_info.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_extract_program_information.sh
# Edit RESULT / OUT and the args below; every arg is listed, defaults included.
# <perturbation_path_base>_<COND>.txt must exist for every level of obs[--categorical_key],
# and no other files may match the base.
# --programs takes space-separated ids (e.g. 1 2 3); leave it out to use all programs.
# For CRT perturbation results use --log2fc_key approx_log2FC.
# Each bundle gets a regulator_gene block: every regulator (top --top_regulator per condition,
# merged) x the --top_gene program genes + --top_unique_gene distinctive genes; OmniPath and the
# literature steps investigate exactly these pairs.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/0.Prepare
RESULT=/path/to/out_dir/run_name
OUT=$RESULT/Interpretation/AGeneTic_test

python $SCRIPT_DIR/Extract_program_information.py \
  --mdata_path $RESULT/Inference/adata/cNMF_60_0_2.h5mu \
  --GO_path $RESULT/Evaluation/60_0_2/60_GO_term_enrichment.txt \
  --perturbation_path_base $RESULT/Evaluation/60_0_2/60_perturbation_association_results \
  --out_dir $OUT \
  --data_key rna \
  --prog_key cNMF \
  --loadings_key loadings \
  --gene_name_key var_names \
  --categorical_key sample \
  --log2fc_key log2FC \
  --cell_type "" \
  --organism human \
  --top_gene 15 \
  --top_unique_gene 5 \
  --membership_top 300 \
  --top_GO 5 \
  --top_regulator 6 \
  --fdr 0.05
