#!/bin/bash
#SBATCH --job-name=driver_gene
#SBATCH -p <partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/driver_gene.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_query_driver_gene.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Run after 1.Search/1.1.Search_literature/search_gene_function.py and 3.Generate/program_label.py.
# For each program and each gene with gene function categorization PDFs, paper-qa answers
# "does <GENE> drive <program_name>, and why?" from that gene's own PDFs only; the answers (verdict,
# reasons, citations per gene) go to <LIT_DIR>/P<k>.json -> literature_evidence.driver_gene.
# paper-qa finds the evidence; Claude (--judge_model) reads it and gives each gene's verdict.
# Relabel loop: fewer than --min_explained genes with a supported / mixed verdict -> Claude rewrites the label
# from the evidence and every gene is asked again (at most --max_rounds times); round r uses up to
# --pdf_schedule[r] PDFs per gene (3 5 7), downloading more first. The best round's label REPLACES
# <LABEL_DIR>/P<k>.json (every label tried kept in its label_history); --max_rounds 0 never relabels.
# Answers are cached per gene in <LIT_DIR>/driver_gene_paperqa_cache/; add --overwrite to re-run paper-qa.
# OPENAI_API_KEY (paper-qa) and ANTHROPIC_API_KEY (judge, relabel, gene function summaries) go in C_Annotation/.env.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/2.Evidence_curation
RESULT=/path/to/out_dir/run_name
LIT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
LABEL_DIR=$RESULT/Interpretation/AGeneTic_test/Program_label
OVERWRITE=""   # set to --overwrite to re-run paper-qa

python $SCRIPT_DIR/query_driver_gene.py \
  --lit_dir $LIT_DIR \
  --label_dir $LABEL_DIR \
  --cache_dir $LIT_DIR/driver_gene_paperqa_cache \
  --config $SCRIPT_DIR/config.yaml \
  --programs 1 \
  --max_citations 3 \
  --pdf_schedule 3 5 7 \
  --judge_model claude-sonnet-5 \
  --judge_effort medium \
  --min_explained 3 \
  --max_rounds 3 \
  --explain_verdicts supported mixed \
  --label_model claude-sonnet-5 \
  --label_effort medium \
  --top_regulator 6 \
  $OVERWRITE
