#!/bin/bash
#
# Precompute PCA/UMAP into a cNMF h5mu so Stage 3b/3c plotting jobs skip it.
#
# Usage (self-sizing -- run it, do NOT sbatch it):
#   ./run_add_umap.sh <h5mu_path> [extra add_umap_to_h5mu.py flags...]
#
# Run directly and it measures the h5mu, derives --mem/--time/--cpus-per-task, and
# re-submits itself with those overrides. Do NOT prefix it with sbatch -- if you do,
# that job just re-submits a correctly sized one and exits, costing an extra hop.
# To pin resources yourself, set ADD_UMAP_INJOB to skip the sizing path entirely:
#   ADD_UMAP_INJOB=1 sbatch --mem=64G --time=06:00:00 run_add_umap.sh <h5mu>
# The static #SBATCH directives below only apply on that manual path; they are
# sized for the largest jamboree input.
#
# Sizing basis (measured 2026-09-15). Peak RSS is driven by the expression matrix
# and, in the dense case, by the temporary numpy allocates for X.var(axis=0) inside
# ensure_umap -- that temporary is the full size of X again, which is why the dense
# Gersbach matrix dominates everything else:
#
#   dataset       rna/X resident                      est. peak
#   Gersbach      46.95 GB dense f32 1.05M x 11138    ~95-115 GB
#   Hon_CM         9.91 GB sparse    nnz 1.24e9        ~30-40 GB
#   Huangfu DE     5.01 GB sparse    nnz 627M          ~15-20 GB
#   Huangfu ESC    3.37 GB sparse    nnz 422M          ~10-14 GB
#
# --in_place additionally needs free disk for a second copy of the h5mu while it
# writes (48 GB for Gersbach; Oak had ~11 TB free at time of writing).

#SBATCH --job-name=add_umap
#SBATCH --output=/scratch/users/ymo/add_umap_logs/%j.out
#SBATCH --error=/scratch/users/ymo/add_umap_logs/%j.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -o pipefail

SCRIPT_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/perturbNMF-runner/scripts"

MDATA_PATH="$1"
if [ -z "$MDATA_PATH" ]; then
    echo "ERROR: no h5mu given. Usage: ./run_add_umap.sh <h5mu_path> [flags...]"
    exit 2
fi
if [ ! -f "$MDATA_PATH" ]; then
    echo "ERROR: no such file: $MDATA_PATH"
    exit 2
fi

# ---------------------------------------------------------------------------
# Self-sizing submit path.
#
# The guard is the ADD_UMAP_INJOB sentinel this script sets on its own sbatch
# call -- NOT $SLURM_JOB_ID. On Sherlock an OnDemand/code-server session is itself
# a Slurm allocation that exports SLURM_JOB_ID (and SLURM_CPUS_PER_TASK=2), so a
# SLURM_JOB_ID check reads as "already in a batch job" from any interactive shell
# there and silently runs the whole UMAP in that 2-CPU session instead of queuing.
# ---------------------------------------------------------------------------
if [ -z "$ADD_UMAP_INJOB" ]; then
    SIZE_BYTES=$(stat -c %s "$MDATA_PATH")
    SIZE_GB=$(( (SIZE_BYTES + 1073741823) / 1073741824 ))   # ceil to GB

    # 3x the file plus 24GB of headroom covers the dense X.var() temporary, the
    # top-2000-gene copy and the PCA working set. Clamped to [32, 512].
    MEM_GB=$(( 3 * SIZE_GB + 24 ))
    [ "$MEM_GB" -lt 32 ]  && MEM_GB=32
    [ "$MEM_GB" -gt 512 ] && MEM_GB=512

    # Wall time tracks cell count more than bytes (a sparse 1M-cell run is small on
    # disk but still pays a full 1M-point neighbors + UMAP), so the floor is
    # deliberately generous rather than proportional. Clamped to [6, 24] hours.
    TIME_H=$(( 6 + SIZE_GB / 6 ))
    [ "$TIME_H" -gt 24 ] && TIME_H=24

    if [ "$SIZE_GB" -lt 8 ]; then CPUS=8; else CPUS=16; fi

    echo "h5mu:  $MDATA_PATH"
    echo "size:  ${SIZE_GB} GB"
    echo "sized: --mem=${MEM_GB}G --time=${TIME_H}:00:00 --cpus-per-task=${CPUS}"
    mkdir -p /scratch/users/ymo/add_umap_logs
    # Absolute self-reference so this works regardless of the caller's cwd.
    # ADD_UMAP_INJOB=1 is what makes the submitted copy run the workload instead of
    # re-submitting again; --export=ALL keeps the rest of the environment.
    exec sbatch --mem="${MEM_GB}G" \
                --time="${TIME_H}:00:00" \
                --cpus-per-task="${CPUS}" \
                --export=ALL,ADD_UMAP_INJOB=1 \
                "$SCRIPT_DIR/run_add_umap.sh" "$@"
fi

shift

# If the caller passed neither --output nor --in_place, default to in place.
EXTRA_ARGS=("$@")
case " ${EXTRA_ARGS[*]} " in
    *" --output "*|*" --in_place "*) ;;
    *) EXTRA_ARGS+=(--in_place) ;;
esac

START_TIME=$(date +%s)

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE:-unset} MB"
echo "Partition: $SLURM_JOB_PARTITION"
echo "h5mu: $MDATA_PATH"
echo "Size: $(du -h "$MDATA_PATH" | cut -f1)"
echo "Extra args: ${EXTRA_ARGS[*]}"

mkdir -p /scratch/users/ymo/add_umap_logs

echo "Activating conda environment: NMF_Benchmarking"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

echo "Active env: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

python3 "$SCRIPT_DIR/add_umap_to_h5mu.py" \
        --mdata_path "$MDATA_PATH" \
        --data_key rna \
        --prog_key cNMF \
        --n_top_genes 2000 \
        --n_comps 50 \
        "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# Optional / unused flags (add them after the h5mu path on the command line):
#     --output /path/to/new.h5mu     # write a copy instead of replacing the input
#     --in_place                     # replace the input (default if neither is given)
#     --force                        # recompute even if X_umap already exists
#     --data_key rna --prog_key cNMF # override modality names
#     --n_top_genes 2000             # must stay 2000 to match ensure_umap's default,
#                                    # else the plotting jobs recompute anyway
#     --n_comps 50

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: add_umap_to_h5mu.py exited with code $EXIT_CODE"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"
echo "Tip: run 'seff $SLURM_JOB_ID' to right-size the next one."
exit $EXIT_CODE
