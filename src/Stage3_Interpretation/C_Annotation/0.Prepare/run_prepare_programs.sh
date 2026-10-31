#!/bin/bash
#SBATCH --job-name=prepare_programs
#SBATCH -p <partition>
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/prepare_programs.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_prepare_programs.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# First annotation step: one folder per program, <RUN_DIR>/Programs/P<k>/, with genes_top300.tsv,
# regulators_sig.tsv, go_sig.tsv, string_sig.tsv (+ string_genes.json, with the STRING links among the regulators),
# an empty literature.jsonl and plan.md.
# STRING needs network access (about 4 calls per program); add --skip_string without it.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/0.Prepare
RESULT=/path/to/out_dir/run_name
K_DIR=60_0_2                     # Evaluation/<K>_<density threshold> folder of the run
K=60
RUN_DIR=$RESULT/Interpretation/Annotation

python $SCRIPT_DIR/prepare_programs.py \
  --mdata_path $RESULT/Inference/adata/cNMF_${K_DIR}.h5mu \
  --GO_path $RESULT/Evaluation/$K_DIR/${K}_GO_term_enrichment.txt \
  --perturbation_path_base $RESULT/Evaluation/$K_DIR/${K}_perturbation_association_results \
  --run_dir $RUN_DIR \
  --prog_key cNMF \
  --loadings_key loadings \
  --gene_name_key var_names \
  --categorical_key sample \
  --log2fc_key log2FC \
  --cell_type "<cell type>" \
  --species human \
  --n_genes 300 \
  --membership_top 300 \
  --fdr 0.05 \
  --string_background spectra \
  --string_required_score 700 \
  --taxid 9606
