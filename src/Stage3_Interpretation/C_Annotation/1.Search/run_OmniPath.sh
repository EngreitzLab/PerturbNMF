#!/bin/bash
#SBATCH --job-name=omnipath_interact
#SBATCH -p <partition>
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
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
SCRIPT_DIR="$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search/1.0.Search_database/search_gene_interaction"
RESULT="/path/to/output_dir/<run_name>"   # edit: run directory with Interpretation/
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
