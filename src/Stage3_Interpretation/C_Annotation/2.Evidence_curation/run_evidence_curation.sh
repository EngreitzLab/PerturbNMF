#!/bin/bash
#SBATCH --job-name=evidence_curation
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/evidence_curation.%j.log

# Usage: sbatch run_evidence_curation.sh
# Edit RESULT / INFO_DIR and the args below; every arg is listed, defaults included.
# Reads <LIT_DIR>/P<k>.json (from 1.Search/1.1.Search_literature) when present, else <INFO_DIR>/P<k>.json,
# and writes to <LIT_DIR>/P<k>.json (the 1.0 bundles are never modified).
# pairs: OmniPath-found pairs take their evidence from the JSON; not-found pairs with PDFs in
#   <literature_dir>/<A>__<B>/ are classified by paper-qa -> gene_interactions.Literature.
# questions: literature_plan curation questions (llm_query_agent.py) are answered by paper-qa over
#   their query folders (+ <LIT_DIR>/gene_pdfs/<GENE>/ for C7) -> literature_evidence.
# Cached pairs are reused; add --overwrite to re-run paper-qa. OPENAI_API_KEY goes in AGeneTic/.env.
# Leave out --cell_type to use each bundle's cell_type.

set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate geneqa

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/2.Evidence_curation
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info
LIT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info

python $SCRIPT_DIR/curate_evidence.py \
  --info_dir $INFO_DIR \
  --lit_dir $LIT_DIR \
  --literature_dir $LIT_DIR/Literature_search \
  --evidence_dir $RESULT/Interpretation/AGeneTic_test/Evidence_curation \
  --config $SCRIPT_DIR/config.yaml \
  --programs 1 \
  --targets pairs questions
