#!/usr/bin/env python3
"""
SLURM script generator for cNMF benchmarking pipeline.

Generates complete .sh files with SBATCH headers, conda activation,
and the correct Python script invocation for any pipeline stage.
"""

import argparse
import os
import sys
import textwrap

PIPELINE_ROOT = "/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
SKILL_SCRIPTS = os.path.join(PIPELINE_ROOT, ".claude/skills/cnmf-runner/scripts")
EMAIL = "ymo@stanford.edu"

# Stage definitions: script path (relative), conda env, whether GPU is needed
STAGES = {
    "inference-sk": {
        "script": "Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py",
        "conda_env": "sk-cNMF",
        "gpu": False,
    },
    "inference-torch": {
        "script": "Inference/torch-cNMF/Slurm_Version/torch_cnmf_inference_pipeline.py",
        "conda_env": "torch-cNMF",
        "gpu": True,
    },
    "evaluation": {
        "script": "Evaluation/Slurm_Version/cNMF_evaluation_pipeline.py",
        "conda_env": "NMF_Benchmarking",
        "gpu": False,
    },
    "k-selection": {
        "script": "Interpretation/Slurm_Version/cNMF_k_selection.py",
        "conda_env": "torch-cNMF",
        "gpu": False,
    },
    "program-analysis": {
        "script": "Interpretation/Slurm_Version/cNMF_program_analysis.py",
        "conda_env": "NMF_Benchmarking",
        "gpu": False,
    },
    "perturbed-gene": {
        "script": "Interpretation/Slurm_Version/cNMF_perturbed_gene_analysis.py",
        "conda_env": "NMF_Benchmarking",
        "gpu": False,
    },
    "u-test-calibration": {
        "script": "Calibration/Slurm_version/U-test_perturbation_calibration/U-test_perturbation_calibration.py",
        "conda_env": "NMF_Benchmarking",
        "gpu": False,
    },
    "crt-calibration": {
        "script": "Calibration/Slurm_version/CRT/CRT.py",
        "conda_env": "programDE",
        "gpu": False,
    },
    "matched-cell-de": {
        "script": "Calibration/Slurm_version/Matched_cell_programDE/run_matching_de_batch.R",
        "conda_env": "programDE",
        "gpu": False,
        "interpreter": "Rscript",
    },
}


def generate_script(args, passthrough_args):
    """Generate the complete SLURM shell script."""
    stage_info = STAGES[args.stage]
    script_path = os.path.join(PIPELINE_ROOT, stage_info["script"])
    conda_env = stage_info["conda_env"]
    needs_gpu = stage_info["gpu"] or args.gpu

    # Build log directory — each stage logs into its own folder
    if args.log_dir:
        log_dir = args.log_dir
    elif args.stage.startswith("inference"):
        log_dir = f"{args.output_dir}/{args.run_name}/Inference/logs"
    elif args.stage == "evaluation" or args.stage.endswith("-calibration"):
        log_dir = f"{args.output_dir}/{args.run_name}/Evaluation/logs"
    else:
        log_dir = f"{args.output_dir}/{args.run_name}/Plots/logs"

    # GPU constraint lines
    gpu_lines = ""
    if needs_gpu:
        gpu_lines = "#SBATCH --gres=gpu:1                   # Request 1 GPU\n#SBATCH -C GPU_SKU:V100S_PCIE\n"

    # Build the passthrough args string
    # Reconstruct tokens into "--key value" pairs for readable multi-line output
    passthrough_str = ""
    if passthrough_args:
        # Strip leading '--' separator if present
        tokens = list(passthrough_args)
        if tokens and tokens[0] == '--':
            tokens = tokens[1:]

        # Group tokens into lines: each --flag starts a new line with its values
        lines = []
        current_line = []
        for token in tokens:
            if token.startswith('--') and current_line:
                lines.append("        " + " ".join(current_line))
                current_line = [token]
            else:
                current_line.append(token)
        if current_line:
            lines.append("        " + " ".join(current_line))

        if lines:
            passthrough_str = " \\\n".join(lines)
            passthrough_str = " \\\n" + passthrough_str

    # Build script as lines for clean formatting
    lines = [
        '#!/bin/bash',
        '',
        '# SLURM job configuration',
        f'#SBATCH --job-name={args.job_name}',
        f'#SBATCH --output={log_dir}/%j.out',
        f'#SBATCH --error={log_dir}/%j.err',
        f'#SBATCH --partition={args.partition}',
        f'#SBATCH --time={args.time}',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        f'#SBATCH --cpus-per-task={args.cpus}',
        f'#SBATCH --mem={args.mem}',
    ]

    if needs_gpu:
        lines.append('#SBATCH --gres=gpu:1')
        lines.append('#SBATCH -C GPU_SKU:V100S_PCIE')

    lines.extend([
        '',
        '# Email notifications',
        '#SBATCH --mail-type=BEGIN,END,FAIL',
        f'#SBATCH --mail-user={EMAIL}',
        '',
        '# Configuration',
        f'OUT_DIR="{args.output_dir}"',
        f'RUN_NAME="{args.run_name}"',
        f'LOG_DIR="{log_dir}"',
        '',
        '# Store start time',
        'START_TIME=$(date +%s)',
        '',
        '# Print job information',
        'echo "Job started at: $(date)"',
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "Node: $SLURMD_NODENAME"',
        'echo "Working directory: $(pwd)"',
        'echo "CPUs: $SLURM_CPUS_PER_TASK"',
        'echo "Partition: $SLURM_JOB_PARTITION"',
        'echo "Log directory: $LOG_DIR"',
        '',
        '# Create log directory',
        'mkdir -p "$LOG_DIR"',
    ])

    if needs_gpu:
        lines.extend([
            '',
            '# Start resource monitoring',
            'MONITOR_LOG="$LOG_DIR/resource_monitor_${SLURM_JOB_ID}.log"',
            'monitor_resources() {',
            '    while true; do',
            '        echo "$(date \'+%Y-%m-%d %H:%M:%S\')" >> "$MONITOR_LOG"',
            '        echo "=== GPU Usage ===" >> "$MONITOR_LOG"',
            '        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu --format=csv >> "$MONITOR_LOG" 2>/dev/null',
            '        echo "=== Memory ===" >> "$MONITOR_LOG"',
            '        free -h >> "$MONITOR_LOG"',
            '        echo "---" >> "$MONITOR_LOG"',
            '        sleep 30',
            '    done',
            '}',
            'monitor_resources &',
            'MONITOR_PID=$!',
        ])

    lines.extend([
        '',
        f'# Activate conda environment',
        f'echo "Activating conda environment: {conda_env}"',
        'eval "$(conda shell.bash hook)"',
        f'conda activate {conda_env}',
        '',
        'echo "Active env: $CONDA_DEFAULT_ENV"',
        'echo "Python: $(python --version)"',
    ])

    # CRT guide preprocessing (when --preprocess_guide_h5mu is set)
    if args.stage == 'crt-calibration' and getattr(args, 'preprocess_guide_h5mu', None):
        prep_script = os.path.join(SKILL_SCRIPTS, "prepare_guide_data.py")
        guide_output = getattr(args, 'preprocess_guide_output', None) or \
            f"{args.output_dir}/{args.run_name}/Evaluation/guide_data.h5ad"
        assignment_key = getattr(args, 'preprocess_assignment_key', 'target_assignment')
        names_key = getattr(args, 'preprocess_names_key', 'target_names')
        lines.extend([
            '',
            '# Preprocess guide data: convert alternative keys to standard guide_* keys',
            f'GUIDE_H5AD="{guide_output}"',
            'echo "Preparing guide data h5ad..."',
            f'python3 {prep_script} \\',
            f'    --h5mu_path "{args.preprocess_guide_h5mu}" \\',
            f'    --output_path "$GUIDE_H5AD" \\',
            f'    --assignment_key "{assignment_key}" \\',
            f'    --names_key "{names_key}"',
            'echo "Guide preprocessing complete."',
        ])

    lines.extend([
        '',
        f'# Run pipeline',
        f'echo "Running: {args.stage}"',
        f'{stage_info.get("interpreter", "python3")} {script_path}{passthrough_str}',
        '',
        '# Capture exit code',
        'EXIT_CODE=$?',
        'if [ $EXIT_CODE -ne 0 ]; then',
        '    echo "ERROR: Pipeline exited with code $EXIT_CODE"',
        'fi',
    ])

    if needs_gpu:
        lines.extend([
            '',
            '# Stop resource monitoring',
            'kill $MONITOR_PID 2>/dev/null',
            'echo "Final GPU status:"',
            'nvidia-smi 2>/dev/null || echo "GPU not available"',
        ])

    # For inference stages, generate h5mu structure files after pipeline completes
    if args.stage.startswith("inference") and not args.no_structure:
        h5mu_script = os.path.join(SKILL_SCRIPTS, "generate_h5mu_structure.py")
        lines.extend([
            '',
            '# Generate h5mu structure summary files',
            'if [ $EXIT_CODE -eq 0 ]; then',
            f'    ADATA_DIR="{args.output_dir}/{args.run_name}/Inference/adata"',
            '    if [ -d "$ADATA_DIR" ]; then',
            '        echo "Generating h5mu structure files..."',
            f'        python3 {h5mu_script} dummy --adata_dir "$ADATA_DIR"',
            '        echo "h5mu structure generation complete"',
            '    else',
            '        echo "WARNING: adata directory not found at $ADATA_DIR, skipping structure generation"',
            '    fi',
            'else',
            '    echo "Skipping h5mu structure generation due to pipeline error"',
            'fi',
        ])

    lines.extend([
        '',
        '# Elapsed time',
        'END_TIME=$(date +%s)',
        'ELAPSED=$((END_TIME - START_TIME))',
        'HOURS=$((ELAPSED / 3600))',
        'MINUTES=$(((ELAPSED % 3600) / 60))',
        'SECONDS=$((ELAPSED % 60))',
        '',
        'echo "Job completed at: $(date)"',
        'echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"',
        'exit $EXIT_CODE',
    ])

    script = '\n'.join(lines) + '\n'

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM submission script for cNMF pipeline stages"
    )

    # Generator-specific arguments
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=list(STAGES.keys()),
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--job_name', type=str, required=True,
        help='SLURM job name'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--run_name', type=str, required=True,
        help='Run name identifier'
    )
    parser.add_argument(
        '--cpus', type=int, default=10,
        help='CPUs per task (default: 10)'
    )
    parser.add_argument(
        '--mem', type=str, default='64G',
        help='Memory allocation (default: 64G)'
    )
    parser.add_argument(
        '--time', type=str, default='10:00:00',
        help='Time limit HH:MM:SS (default: 10:00:00)'
    )
    parser.add_argument(
        '--partition', type=str, default='engreitz,owners',
        help='SLURM partition(s) (default: engreitz,owners)'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Request GPU resources (auto-set for torch stages)'
    )
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='Custom log directory (default: <output_dir>/<run_name>/logs)'
    )
    parser.add_argument(
        '--script_output_path', type=str, default=None,
        help='Where to write the generated .sh file (default: stdout)'
    )
    parser.add_argument(
        '--no_structure', action='store_true',
        help='Skip h5mu structure file generation after inference'
    )

    # CRT guide preprocessing arguments
    parser.add_argument(
        '--preprocess_guide_h5mu', type=str, default=None,
        help='(CRT only) Path to h5mu file for guide data extraction. '
             'When set, the generated script includes a preprocessing step '
             'that converts target_*/guide_* keys to the standard format '
             'expected by CRT.py.'
    )
    parser.add_argument(
        '--preprocess_guide_output', type=str, default=None,
        help='(CRT only) Output path for the preprocessed guide h5ad '
             '(default: <save_dir>/guide_data.h5ad if --save_dir is in '
             'passthrough args, otherwise <output_dir>/<run_name>/Evaluation/guide_data.h5ad)'
    )
    parser.add_argument(
        '--preprocess_assignment_key', type=str, default='target_assignment',
        help='(CRT only) Source obsm key for guide assignment matrix (default: target_assignment)'
    )
    parser.add_argument(
        '--preprocess_names_key', type=str, default='target_names',
        help='(CRT only) Source uns key for guide/target names (default: target_names)'
    )

    # Parse only known args; the rest are passthrough to the pipeline script
    args, passthrough = parser.parse_known_args()

    script = generate_script(args, passthrough)

    if args.script_output_path:
        os.makedirs(os.path.dirname(args.script_output_path), exist_ok=True)
        with open(args.script_output_path, 'w') as f:
            f.write(script)
        os.chmod(args.script_output_path, 0o755)
        print(f"Script written to: {args.script_output_path}")
    else:
        print(script)


if __name__ == '__main__':
    main()
