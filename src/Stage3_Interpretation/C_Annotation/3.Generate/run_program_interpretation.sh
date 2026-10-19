#!/bin/bash
#SBATCH --job-name=program_interpretation
#SBATCH -p <partition>
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/program_interpretation.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_interpretation.sh   (one API call per program)
# Edit RESULT and the args below; every arg is listed, defaults included.
# After run_program_label.sh (and 2.Evidence_curation/run_query_driver_gene.sh, whose verdicts it uses):
# reads <INFO_DIR>/P<k>.json + the program's label and writes a Claude-generated interpretation of that label
# to <OUT_DIR>/P<k>.json: confounder rule-outs, layered interpretation (upstream trigger, co-regulation
# mechanism, cellular output), modules and coherence. Genes / regulators / PMIDs outside the evidence are
# removed. Cached files are reused (redone when the label changed); add --overwrite to redo them.
# Drop the --motif_* lines to leave out the TF motif evidence. Then run_program_viewer.sh.
# ANTHROPIC_API_KEY comes from C_Annotation/.env.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/3.Generate
RESULT=/path/to/out_dir/run_name
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
K_DIR=60_0_2                     # Evaluation/<K>_<density threshold> folder of the run
K=60
OVERWRITE=""   # set to --overwrite to redo programs that already have an interpretation

python $SCRIPT_DIR/program_interpretation.py \
  --info_dir $INFO_DIR \
  --label_dir $RESULT/Interpretation/AGeneTic_test/Program_label \
  --out_dir $RESULT/Interpretation/AGeneTic_test/Program_interpretation \
  --programs 1 \
  --top_regulator 6 \
  --motif_enrichment_path $RESULT/Evaluation/$K_DIR/${K}_motif_enrichment.txt \
  --candidate_tfs_path $RESULT/Evaluation/$K_DIR/${K}_candidate_tfs.txt \
  --model claude-sonnet-5 \
  --effort medium \
  --max_tokens 8000 \
  $OVERWRITE
