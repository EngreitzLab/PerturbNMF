#!/bin/bash
# Call TF-motif hits (FIMO or Fi-NeMo) in promoter / enhancer regions on SLURM.
# All arguments are forwarded to call_motif_hits.py; --n_jobs defaults to the allocated CPUs.
# sbatch runs a spooled copy of this file, so set PIPELINE_SRC to this directory (see example).
#
# Example (promoters from a GTF, hg38, MEME fimo):
#   sbatch --output=<out_dir>/call_motif_hits_%j.log \
#       --export=ALL,PIPELINE_SRC=<repo>/src/Stage2_Evaluation/A_Metrics/Slurm_Version call_motif_hits.sh \
#       --region_type promoter \
#       --gene_annotation <path/to/genes.gtf.gz> \
#       --genome_fasta <path/to/hg38.fa> \
#       --motif_file <path/to/MotifCompendium-Database-Human.meme.txt> \
#       --fimo_binary "$(which fimo)" --n_chunks 64 \
#       --out_dir <out_dir>
# (HOCOMOCO v11, the Schnitzler et al. 2024 database: <path/to/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme>;
#  run_motif_enrichment.py then needs --motif_db hocomoco_v11)
# Needs CONDA_BASE (your conda install) in the environment.
#
#SBATCH --job-name=call_motif_hits
#SBATCH --partition=<partition>
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

set -euo pipefail

SCRIPT_DIR="${PIPELINE_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda install}"

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} started $(date)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate NMF_Benchmarking
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"

python "${SCRIPT_DIR}/call_motif_hits.py" --n_jobs "${SLURM_CPUS_PER_TASK:-1}" "$@"
echo "Job finished $(date)"
