#!/bin/bash
#SBATCH --job-name=generif_lit
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/generif_lit.%j.out
#SBATCH --error=logs/generif_lit.%j.err

# Usage: sbatch run_search_generif.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Reads <INFO_DIR>/P<k>.json (1.0.Search_database output, needs gene_info.NCBI.generif_pmids) and writes
# <OUT_DIR>/P<k>.json with gene_info.GeneRIF on every gene; P<k>.json already in OUT_DIR are extended.
# DOWNLOAD_PDFS=--download_pdfs saves open-access PDFs to <OUT_DIR>/gene_pdfs/<GENE>/<PMID>.pdf.
# Per-gene results are cached in <OUT_DIR>/gene_cache/; add --overwrite to re-summarize.
# Needs ANTHROPIC_API_KEY (and optionally NCBI_API_KEY / NCBI_EMAIL / UNPAYWALL_EMAIL) in AGeneTic/.env.
set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate geneqa

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/1.Search/1.1.Search_literature
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info
OUT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
DOWNLOAD_PDFS="--download_pdfs"   # set to "" to skip PDF downloads
OVERWRITE=""                      # set to --overwrite to ignore gene_cache/

mkdir -p $OUT_DIR/logs
exec > >(tee -a $OUT_DIR/logs/generif_lit.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $OUT_DIR/logs/generif_lit.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/search_generif.py \
  --info_dir $INFO_DIR \
  --out_dir $OUT_DIR \
  --programs 1 \
  --context_terms endothelial HUVEC HAEC aorta \
  --top_n 5 \
  --pdf_chars 15000 \
  --model claude-sonnet-5 \
  --effort medium \
  --max_tokens 16000 \
  $DOWNLOAD_PDFS $OVERWRITE
