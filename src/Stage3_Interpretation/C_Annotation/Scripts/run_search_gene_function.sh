#!/bin/bash
#SBATCH --job-name=gene_function_lit
#SBATCH -p <partition>
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/gene_function_lit.%j.out
#SBATCH --error=logs/gene_function_lit.%j.err

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_search_gene_function.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Gene function categorization: for every gene of each program, PubMed is searched for papers on the gene's
# function, N_PDFS open-access PDFs are downloaded to <OUT_DIR>/gene_function_pdfs/<GENE>/<PMID>.pdf, and one
# LLM call per gene summarizes them and assigns 1-3 function categories. Reads <INFO_DIR>/P<k>.json (1.0 output)
# and writes <OUT_DIR>/P<k>.json with gene_info.Gene_function_categorization; P<k>.json already in OUT_DIR are extended.
# Per-gene results are cached in <OUT_DIR>/gene_function_cache/; add --overwrite to re-categorize.
# Needs ANTHROPIC_API_KEY (and optionally NCBI_API_KEY / NCBI_EMAIL / UNPAYWALL_EMAIL) in AGeneTic/.env.
set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search/1.1.Search_literature
RESULT=/path/to/out_dir/run_name
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info
OUT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
OVERWRITE=""   # set to --overwrite to ignore gene_function_cache/

mkdir -p $OUT_DIR/logs
exec > >(tee -a $OUT_DIR/logs/gene_function_lit.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $OUT_DIR/logs/gene_function_lit.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/search_gene_function.py \
  --info_dir $INFO_DIR \
  --out_dir $OUT_DIR \
  --programs 1 \
  --gene_lists program_genes distinctive_genes regulators \
  --context_terms \
  --n_pdfs 3 \
  --pool_size 20 \
  --max_aliases 3 \
  --function_terms function role "regulat*" mechanism \
  --pdf_chars 15000 \
  --model claude-sonnet-5 \
  --effort medium \
  --max_tokens 16000 \
  $OVERWRITE
