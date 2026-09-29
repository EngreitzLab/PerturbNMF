#!/bin/bash
#SBATCH --job-name=omnipath_interact
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/omnipath_interact.%j.log

# Usage: sbatch run_OmniPath.sh
# Edit RESULT / BUNDLE_DIR and the args below; every arg is listed, defaults included.
# Results are added as "gene_interactions": {"OmniPath": {...}} to <out_dir>/P<k>.json
# (default <BUNDLE_DIR>/../Gene_info_extended_PerturbNMF_Info); other content is kept.
# The pairs tested are each bundle's regulator_gene list (Extract_program_information.py: every
# regulator x program/distinctive gene); --top_gene / --top_regulator / --include_unique /
# --top_unique_gene only apply to older bundles without that list.
# Run search_MyGene.py, search_NCBI.py, search_UniProt.py first so gene aliases can be tried.
# Programs already searched with the same params are skipped; add --overwrite to re-query.
# --programs (required) takes space-separated ids (e.g. 1 2 3); only those P<k>.json are read.

set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate NMF_Benchmarking

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/1.Search/1.0.Search_database/search_gene_interaction
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
BUNDLE_DIR=$RESULT/Interpretation/AGeneTic_test/PerturbNMF_Info

python $SCRIPT_DIR/search_OmniPath.py \
  --bundle_dir $BUNDLE_DIR \
  --out_dir $RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info \
  --programs 1 \
  --taxid 9606 \
  --top_gene 15 \
  --top_regulator 6 \
  --include_unique \
  --top_unique_gene 8 \
  --datasets omnipath pathwayextra kinaseextra ligrecextra collectri dorothea tf_target mirnatarget lncrna_mrna tf_mirna small_molecule \
  --max_pmids 20
