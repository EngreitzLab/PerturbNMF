# Running on SLURM

Each stage has a `Slurm_Version/` (or `Slurm_version/`) folder with a Python entry point and a
matching `.sh` runner. Before submitting a runner:

1. `export PIPELINE_ROOT=/path/to/PerturbNMF`
2. Replace `<partition>` and `<your_email>` in the `#SBATCH` header, or pass
   `--partition` / `--email` to the runner skill's script generator.
3. Check the `--mem`, `--time` and `--gres` requests for your data size.

```bash
sbatch src/Stage1_Inference/torch-cNMF/Slurm_Version/torch_cnmf_inference_pipeline.sh
```

Logs go to `<out_dir>/<run_name>/{Inference,Evaluation,Plots}/logs/`. Each job saves its
config as `config_<SLURM_JOB_ID>.yml`.

## Guided runs with Claude Code

The repo includes a [`perturbNMF-runner`](https://github.com/EngreitzLab/PerturbNMF/tree/main/.claude/skills/perturbNMF-runner)
Claude Code skill. It checks your inputs, suggests resource requests, and writes and submits the
SLURM scripts. Open the repo in Claude Code and ask it to "run PerturbNMF".
