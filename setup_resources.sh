#!/bin/bash
# Setup script: creates the Evaluation/Resources directory
# Run this once after cloning the repo on Sherlock
RESOURCES_SRC="/oak/stanford/groups/engreitz/Users/ymo/Tools/cNMF_benchmarking/cNMF_benchmarking_pipeline/Evaluation/Resources"
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
    echo "Please update RESOURCES_SRC in this script to point to your Resources directory."
    exit 1
fi
