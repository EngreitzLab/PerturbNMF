# Annotation & Excel Summary Stages

> **All-flags convention (mandatory for every generated `.sh`):** When invoking `generate_slurm.py`, list active flags first, then `---COMMENTED---`, then every remaining flag for this stage from `references/parameter-catalog.md` (annotation / excel-summary parameter sections) with a sensible default/example value. The generator emits unused flags as `#     --flag value` lines below the python command so the user can toggle them later. See `SKILL.md` Step 5.

---

## Annotation (Stage 3d)

**Conda**: `progexplorer`

LLM-driven gene program annotation. Runs the PerturbNMF Annotation pipeline which extracts top genes per program, queries STRING for protein interactions, mines literature, builds prompts, and submits to an LLM for annotation.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Path to pipeline config YAML (see `src/Stage3_Interpretation/C_Annotation/configs/pipeline_config.yaml` for template) |

The config YAML specifies: input spectra file, output directory, LLM model, STRING parameters, and literature mining settings. Any of the YAML keys can also be overridden on the command line — see `references/parameter-catalog.md` (Annotation section) for the full list.

### Common CLI overrides (skip the YAML for one-offs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gene-loading` | (from YAML) | Path to gene loading matrix CSV |
| `--celltype-enrichment` | None | Raw cell-type enrichment CSV (auto-summarized if provided) |
| `--output-dir` | (from YAML) | Output directory for all results |
| `--regulator-file` | None | SCEPTRE regulator results CSV |
| `--topics` | all | Comma-separated topic IDs to process (e.g. `2,6,33`) |
| `--species` | `10090` | NCBI taxonomy ID (`10090` = mouse, `9606` = human) |
| `--keyword` | (from YAML) | PubMed search keyword for tissue/cell type |
| `--annotation-role` | (from YAML) | Specialist role used in the LLM prompt header |
| `--annotation-context` | (from YAML) | Dataset/cell-type description for the prompt header |
| `--top-positive-regulators` | (from YAML) | Number of positive regulators per program |
| `--top-negative-regulators` | (from YAML) | Number of negative regulators per program |
| `--regulator-significance-threshold` | (from YAML) | Adjusted p-value cutoff when regulator file lacks a `significant` column |
| `--start-from` | None | Resume from a step: `string_enrichment`, `literature_fetch`, `batch_prepare`, `batch_submit`, `parse_results`, `html_report` |
| `--stop-after` | None | Stop after a step (same choices as `--start-from`) |
| `--restart-from` | None | Re-run from the specified step (overwrites later state) |
| `--gcs-prefix` | None | GCS prefix for batch results (for resuming at `parse_results`) |
| `--no-resume` | off | Disable resume/caching; re-query all APIs |
| `--wait` | off | Wait for LLM batch completion (default: submit and exit; resume later) |
| `--force-restart` | off | Ignore existing state and restart pipeline (overwrites prior output) |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 32G
- Time: 1-2h (depends on number of programs and LLM response time)

---

## Literature Search (optional companion to Annotation)

**Conda**: `progexplorer`

Mines PubMed/PubTator for evidence supporting the program annotations produced by the Annotation stage. Run after Annotation if you want literature citations attached to each program.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--excel` | Input Excel with one row per program (typically the Annotation HTML report's source workbook) |
| `--output-dir` | Output directory for per-program literature pages |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--programs` | all | Comma-separated program IDs (e.g. `2,6,33,34`) |
| `--interactions` | (built-in 17-verb list) | Comma-separated interaction verbs used to formulate queries |
| `--domain-keywords` | (built-in vascular set) | Comma-separated domain keywords for evidence scoring |
| `--max-papers` | `30` | Max papers per program |
| `--max-pubtator-results` | `50` | Max results per PubTator query |
| `--max-llm-queries` | `8` | Max LLM-generated queries per program |
| `--llm-provider` | `stanford` | One of `anthropic`, `stanford`, `openai`, `deepseek`, `gemini` |
| `--llm-model` | None (provider default) | LLM model name |
| `--llm-max-tokens` | `4096` | Max tokens for LLM output |
| `--semantic-check` | off | Enable LLM semantic verification (costs tokens) |
| `--resume` / `--no-resume` | resume on | Enable/disable resume/caching |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 32G
- Time: 1-3h (depends on `--max-papers` and LLM throughput)

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
