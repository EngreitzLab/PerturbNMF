#!/bin/bash
set -euo pipefail

# SLURM job configuration
#SBATCH --job-name=lit_search
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=engreitz
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

START_TIME=$(date +%s)

# Print job and system information for debugging
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Working directory: $(pwd)"

# Configuration
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIT_SEARCH_DIR="$(cd "$SLURM_DIR/.." && pwd)"
OUTPUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/lit_search_output"
LOG_DIR="$OUTPUT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate progexplorer

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Load .env for API keys
if [ -f "$LIT_SEARCH_DIR/.env" ]; then
    export $(grep -v '^#' "$LIT_SEARCH_DIR/.env" | xargs)
fi

# Record start time and initial memory
SCRIPT_START_TIME=$(date)
echo "Script start time: $SCRIPT_START_TIME"
echo "Initial memory usage:"
free -h

# Run the Python script
echo "Running Literature Search pipeline..."
python3 "$SLURM_DIR/run_literature_search.py" \
        --excel "/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/tests/Script/Stage3_Interpretation/C_Annotation/Data/test_programs.xlsx" \
        --output-dir "$OUTPUT_DIR" \
        --programs "2,6,33,34" \
        --llm-provider "anthropic" \
        --llm-model "claude-sonnet-4-5-20250929" \
        --max-papers 30 \
        --interactions "regulates,induces,promotes,inhibits,suppresses,activates,binds,modulates" \
        --domain-keywords "angiogenesis,permeability,barrier,inflammation,proliferation,migration,sprouting,hypoxia,metabolism" \
        #--semantic-check \
        #--no-resume \

# Cleanup function — runs on both success and failure
cleanup() {
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    SCRIPT_END_TIME=$(date)
    DURATION=$((END_TIME - START_TIME))

    # Final resource summary
    echo "========================================="
    echo "EXECUTION SUMMARY"
    echo "========================================="
    if [ $EXIT_CODE -ne 0 ]; then
        echo "*** PIPELINE FAILED with exit code $EXIT_CODE ***"
    fi
    echo "Script start time: $SCRIPT_START_TIME"
    echo "Script end time: $SCRIPT_END_TIME"
    echo "Total execution time: ${DURATION} seconds ($(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s)"
    echo "Final memory usage:"
    free -h

    # Peak memory usage from SLURM
    echo "SLURM reported peak memory usage: $(sacct -j ${SLURM_JOB_ID:-0} --format=MaxRSS --noheader | head -1 | tr -d ' ')" 2>/dev/null || echo "SLURM memory stats not available"

    echo "Output directory: $OUTPUT_DIR"
    echo "Job completed at: $(date)"
    exit $EXIT_CODE
}
trap cleanup EXIT
