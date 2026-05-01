# Annotation & Excel Summary Stages

---

## Annotation (Stage 3d)

**Conda**: `progexplorer`

LLM-driven gene program annotation. Runs the PerturbNMF Annotation pipeline which extracts top genes per program, queries STRING for protein interactions, mines literature, builds prompts, and submits to an LLM for annotation.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Path to pipeline config YAML (see `src/Stage3_Interpretation/C_Annotation/configs/pipeline_config.yaml` for template) |

The config YAML specifies: input spectra file, output directory, LLM model, STRING parameters, and literature mining settings.

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 32G
- Time: 1-2h (depends on number of programs and LLM response time)

---

## Excel Summarization (Stage 3e)

**Conda**: `NMF_Benchmarking`

Compiles all evaluation and calibration results into a single Excel workbook for easy review.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--mdata_path` | Path to .h5mu file |
| `--output_path` | Output path for the Excel file |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 2
- Memory: 32G
- Time: 30min
