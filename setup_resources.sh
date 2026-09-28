#!/bin/bash
# Setup script: populates src/Stage2_Evaluation/Resources (large reference data, not tracked in git).
# Run this once after cloning the repo.
#
# Usage:
#   RESOURCES_SRC=/path/to/Resources ./setup_resources.sh
#
# RESOURCES_SRC must be an existing directory containing the evaluation reference files, e.g.:
#   OpenTargets_L2G_Filtered.csv.gz  - Open Targets locus-to-gene (L2G) GWAS associations (trait enrichment)
#   hocomoco_meme.meme               - HOCOMOCO TF motifs in MEME format (motif enrichment)
#   hg38.fa                          - hg38 genome FASTA (motif enrichment; e.g. from UCSC hgdownload)
# The directory is symlinked into the repo so multiple checkouts can share one copy.
RESOURCES_SRC="${RESOURCES_SRC:?set RESOURCES_SRC to a directory containing the Resources files (see header)}"
RESOURCES_DST="$(dirname "$0")/src/Stage2_Evaluation/Resources"

if [ -d "$RESOURCES_DST" ] || [ -L "$RESOURCES_DST" ]; then
    echo "Resources already exist at $RESOURCES_DST"
    exit 0
fi

if [ -d "$RESOURCES_SRC" ]; then
    ln -s "$RESOURCES_SRC" "$RESOURCES_DST"
    echo "Symlinked $RESOURCES_DST -> $RESOURCES_SRC"
else
    echo "ERROR: Source not found at $RESOURCES_SRC"
    echo "Set RESOURCES_SRC to the directory holding your Resources files."
    exit 1
fi
