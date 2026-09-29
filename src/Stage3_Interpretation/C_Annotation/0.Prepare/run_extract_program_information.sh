#!/bin/bash
#SBATCH --job-name=extract_prog_info
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/extract_prog_info.%j.log

# Usage: sbatch run_extract_program_information.sh
# Edit RESULT / OUT and the args below; every arg is listed, defaults included.
# <perturbation_path_base>_<COND>.txt must exist for every level of obs[--categorical_key],
# and no other files may match the base.
# --programs takes space-separated ids (e.g. 1 2 3); leave it out to use all programs.
# For CRT perturbation results use --log2fc_key approx_log2FC.
# Each bundle gets a regulator_gene block: every regulator (top --top_regulator per condition,
# merged) x the --top_gene program genes + --top_unique_gene distinctive genes; OmniPath and the
# literature steps investigate exactly these pairs.

set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate NMF_Benchmarking

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/0.Prepare
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
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
  --categorical_key type \
  --log2fc_key log2FC \
  --cell_type "teloHAEC aortic endothelial cell" \
  --organism human \
  --top_gene 15 \
  --top_unique_gene 8 \
  --membership_top 300 \
  --top_GO 10 \
  --top_regulator 6 \
  --fdr 0.05
