#!/bin/bash
#SBATCH --job-name=uniprot_extend
#SBATCH -p <partition>
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/uniprot_extend.%j.out
#SBATCH --error=logs/uniprot_extend.%j.err

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_UniProt.sh
# Edit RESULT / BUNDLE_DIR and the args below; every arg is listed, defaults included.
# Output goes to <BUNDLE_DIR>/../Gene_info_extended_PerturbNMF_Info unless --out_dir is given; P<k>.json
# already there (from MyGene.py / NCBI.py / UniProt.py) are extended, not replaced.
# Genes that already have this source's info are skipped; add --overwrite to re-query them.
# --programs (required) takes space-separated ids (e.g. 1 2 3); only those P<k>.json are read.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search/1.0.Search_database/search_gene
RESULT=/path/to/out_dir/run_name
BUNDLE_DIR=$RESULT/Interpretation/AGeneTic_test/PerturbNMF_Info
OVERWRITE=""   # default off; set to --overwrite to re-query genes that already have UniProt info
OUT_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info

# copy stdout/stderr into the output folder too (Slurm still writes logs/ in the submit dir)
mkdir -p $OUT_DIR/logs
exec > >(tee -a $OUT_DIR/logs/uniprot_extend.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $OUT_DIR/logs/uniprot_extend.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/search_UniProt.py \
  --bundle_dir $BUNDLE_DIR \
  --out_dir $OUT_DIR \
  --programs $(seq 1 60) \
  --taxid 9606 \
  --batch_size 100 \
  $OVERWRITE
