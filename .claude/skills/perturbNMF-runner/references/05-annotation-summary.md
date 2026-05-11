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

Compiles all evaluation and calibration results into a single multi-sheet Excel workbook. Generates a Jupyter notebook tailored to the project, then executes it.

**Source library**: `src/Stage3_Interpretation/B_Summarization/src/Compile_excel_sheet.py`
**Reference notebook**: `src/Stage3_Interpretation/B_Summarization/JupterNote_Version/cNMF_compile_excel_table.ipynb`

### How it works

There is no standalone CLI script for this stage. Instead, generate a project-specific Jupyter notebook (adapted from the reference notebook above) and execute it via `jupyter nbconvert --execute`. The notebook calls library functions from `Compile_excel_sheet.py`.

### Required parameters (collected interactively)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `out_dir` | Parent directory containing the run folder | `/path/to/project/Result` |
| `run_name` | Run name identifier | `030726_20iter_5KHVG_torch_halsvar_batch_e7` |
| `eval_dir_name` | Name of the evaluation results subdirectory | `Evaluation` or `Eval` (varies by project) |
| `components` | List of K values to compile | `[50]` |
| `sel_threshs` | List of density thresholds | `[0.2]` or `[2.0]` |
| `Sample` | Sample/condition labels for perturbation files | `['WTC']` or `['d0','d1','d2','d3']` |
| `categorical_key` | obs column for sample/condition grouping | `'batch'` or `'timepoint'` |
| `perturbation_file_name` | Prefix for perturbation result files | `'perturbation_association_results'` or `'CRT'` |
| `non_targeting_key` | Guide target labels used as negative controls | `['non-targeting']` |
| `effect_size` | Column name for effect size in perturbation results | `'log2FC'` |

### Optional parameters (with defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_gene` | `300` | Number of top genes per program in loadings sheet |
| `prog_key` | `'cNMF'` | Modality key for cNMF programs in MuData |
| `data_key` | `'rna'` | Modality key for RNA expression in MuData |
| `guide_targets_key` | `'guide_targets'` | uns key for guide target names |

### Key path conventions

Evaluation results are expected at:
```
{out_dir}/{run_name}/{eval_dir_name}/{K}_{thresh}/
├── {K}_GO_term_enrichment.txt
├── {K}_geneset_enrichment.txt
├── {K}_trait_enrichment.txt
├── {K}_{perturbation_file_name}_{sample}.txt   (one per sample)
├── {K}_categorical_association_results.txt
└── {K}_Explained_Variance.txt
```

Where `{thresh}` = `str(sel_thresh).replace('.', '_')` (e.g., `0.2` → `0_2`, `2.0` → `2_0`).

**IMPORTANT**: The `load_simple_sheets()` helper in `Compile_excel_sheet.py` hardcodes `Evaluation/` as the subdirectory name. If the project uses a different name (e.g., `Eval/`), do NOT use `load_simple_sheets()`. Instead, construct paths manually and call individual compile functions (`Compile_GO_sheet`, `Compile_Perturbation_sheet`, etc.) directly with explicit paths.

### Guide data handling

- If guide data (`guide_names`, `guide_targets`, `guide_assignment`) is already embedded in the h5mu: no extra loading needed.
- If guide data lives in a separate file: load it and assign via a helper function (see reference notebook Step 2).

### Sample vs categorical_key alignment

`Compile_Summary_sheet()` uses `Sample` for the "Automatic Timepoint" column (maps `Mean program score {samp}` columns). These column names come from `get_program_info_Summary_cols()`, which uses `categorical_key` values. So:

- If `Sample` matches `categorical_key` values (e.g., both are timepoints): pass `Sample` directly.
- If `Sample` differs from `categorical_key` (e.g., `Sample=['WTC']` but `categorical_key='batch'`): pass `batch_values` as `Sample` to `Compile_Summary_sheet` for "Automatic Timepoint" to work. The perturbation-specific columns use `df_Perturbation['Sample'].unique()` internally, so they still reflect the actual perturbation samples.

### Output

One Excel file per (K, threshold):
```
{out_dir}/{run_name}/Interpretation/Summary_table/{K}_{thresh}/cNMF_{K}_{thresh}.xlsx
```

### Output sheets

| Sheet | Description |
|-------|-------------|
| **Summary** | One row per program: top genes, enrichment highlights, perturbation hit counts, mean scores per condition |
| **Program Loadings** | Long-format gene loading scores with gene descriptions (via MyGene API) |
| **Targets Summary** | Per-target aggregated perturbation stats: expression, cell counts, significant programs, correlations |
| **Sample Association** | Kruskal-Wallis + Dunn posthoc p-values per program |
| **Perturbation Association** | Full Mann-Whitney U test results (split across sheets if >1M rows) |
| **Significant Regulators Only** | Perturbation Association filtered to adj_pval < 0.05 |
| **Trait Enrichment** | GWAS trait enrichment via Fisher exact test (Open Targets L2G) |
| **GO Term Enrichment** | GO Biological Process 2023 enrichment |
| **Geneset Enrichment** | Reactome 2022 pathway enrichment |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 2
- Memory: 32G
- Time: 1h (MyGene API queries for gene annotations are the bottleneck)
