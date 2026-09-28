#!/bin/bash
# Program TF-motif enrichment + candidate TFs for a PerturbNMF run on SLURM (run_motif_enrichment.py).
# All arguments are forwarded; --n_jobs defaults to the allocated CPUs.
# sbatch runs a spooled copy of this file, so set PIPELINE_SRC to this directory (see example).
#
# Example (hg38 run; FIMO promoters from a GTF + element-gene link enhancers; MEME fimo):
#   sbatch --output=<out_dir>/<run_name>/Evaluation/logs/motif_%j.log \
#       --export=ALL,PIPELINE_SRC=<repo>/src/Stage2_Evaluation/A_Metrics/Slurm_Version run_motif_enrichment.sh \
#       --out_dir <out_dir> --run_name <run_name> --K <K> --sel_threshs 0.2 \
#       --enhancer_links <path/to/links.bedpe.gz> \
#       --genome_fasta <path/to/hg38.fa> --gene_annotation <path/to/genes.gtf.gz> \
#       --motif_file <path/to/MotifCompendium-Database-Human.meme.txt> \
#       --fimo_binary "$(which fimo)"
# Add Fi-NeMo hits (ENCODE ChromBPNet; tars are extracted into the hit cache):
#       --motif_source both --finemo_instances <ENCFF...instances.tar.gz> --finemo_report <ENCFF...report.tar.gz>
# Defaults: --motif_db motifcompendium (--motif_db hocomoco_v11 = the Schnitzler et al. 2024 setup);
#           --genome_fasta / --gene_annotation / --motif_file fall back to $PERTURBNMF_GENOME_FASTA,
#           $PERTURBNMF_GENE_ANNOTATION, $PERTURBNMF_MOTIFCOMPENDIUM_MEME (or $PERTURBNMF_HOCOMOCO_V11_MEME).
# Needs CONDA_BASE (your conda install) in the environment.
#
#SBATCH --job-name=motif_enrichment
#SBATCH --partition=<partition>
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="${PIPELINE_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda install}"

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} started $(date)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate NMF_Benchmarking
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"

python "${SCRIPT_DIR}/run_motif_enrichment.py" --n_jobs "${SLURM_CPUS_PER_TASK:-1}" "$@"
echo "Job finished $(date)"
