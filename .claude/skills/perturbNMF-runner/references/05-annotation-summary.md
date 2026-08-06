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

Compiles Stage 1 (`.h5mu`) and Stage 2 (Evaluation) outputs into a single multi-sheet
Excel workbook for **one `(K, sel_thresh)` per job**. There is now a standalone CLI
wrapper — generate the `.sh` with `generate_slurm.py --stage excel-summary` like any
other stage (no notebook required). Submit one job per K to cover multiple K values.

**Script**: `src/Stage3_Interpretation/B_Summarization/Slurm_Version/cNMF_excel_summary.py`
**Source library**: `src/Stage3_Interpretation/B_Summarization/src/Compile_excel_sheet.py`
**Reference notebook (interactive equivalent)**: `src/Stage3_Interpretation/B_Summarization/JupterNote_Version/cNMF_compile_excel_table.ipynb`

### Required parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--out_dir` | Output root directory (contains `{run_name}/`) | `/path/to/project/Result` |
| `--run_name` | Run name identifier | `030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50` |
| `--K` | Number of components (single K) | `50` |

### Commonly set parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sel_thresh` | `0.2` | Density threshold (`0.2` → `0_2`, `2.0` → `2_0`) |
| `--Sample` | `D0 sample_D1 sample_D2 sample_D3` | Condition/sample labels (e.g. `D0 D1 D2 D3`, `WTC`) |
| `--categorical_key` | `sample` | obs column for sample/condition grouping (e.g. `batch`, `timepoint`) |
| `--perturbation_file_name` | `perturbation_association_results` | Perturbation file stem (e.g. `CRT`) |
| `--effect_size` | `log2FC` | Effect-size column (e.g. `approx_log2FC`) |
| `--gene_names_key` | `symbol` | var column with gene symbols |
| `--non_targeting_key` | `non-targeting` | Negative-control target label(s) |

See `references/parameter-catalog.md` (Section 12) for the full flag list, including
`--save_path`, `--mdata_path`, `--num_gene`, the `--prog_key`/`--data_key`/`--guide_targets_key`
keys, `--adjusted_pval_key`, and the per-sheet `--*_Term_key` / `--*_Genes_key` overrides.

### Key path conventions

Reads the per-K evaluation outputs at (paths derived from `--out_dir`/`--run_name`/`--K`/`--sel_thresh`):
```
{out_dir}/{run_name}/Evaluation/{K}_{thresh}/
├── {K}_GO_term_enrichment.txt
├── {K}_geneset_enrichment.txt
├── {K}_trait_enrichment.txt
├── {K}_{perturbation_file_name}_{Sample}.txt   (one per sample)
├── {K}_categorical_association_results.txt
└── {K}_Explained_Variance.txt
```
and the MuData at `{out_dir}/{run_name}/Inference/adata/cNMF_{K}_{thresh}.h5mu`. Where
`{thresh}` = `str(sel_thresh).replace('.', '_')`. Override the auto-derived input with
`--mdata_path` if your h5mu lives elsewhere. (The wrapper calls the individual
`Compile_*` helpers directly, so it does not depend on `load_simple_sheets()`'s hardcoded
`Evaluation/`; for a non-default eval dir, point `--mdata_path` and `--save_path`
explicitly and keep eval files under `Evaluation/` or symlink them.)

### Output

Per job, written to `--save_path` (default `{out_dir}/{run_name}/Interpretation/Summary_table/{K}_{thresh}/`):
- `cNMF_{K}_{thresh}.xlsx` — main multi-sheet workbook
- `Summary_{K}_{thresh}.tsv`, `Program_Loadings_{K}_{thresh}.tsv`, `Targets_Summary_{K}_{thresh}.tsv`
- Sidecars: `specificity_score_{Sample}.txt`, `corr_gene_matrix_{Sample}.txt(.gz)`, `kd_efficiency.txt`, `perturbation_merged_{Sample}(.._significant).txt`
- `config_{SLURM_JOB_ID}.yml`

### Output sheets

| Sheet | Description |
|-------|-------------|
| **Summary** | One row per program: top genes, enrichment highlights, perturbation hit counts, mean scores per condition |
| **Program Loadings** | Long-format gene loading scores with gene descriptions (via MyGene API) |
| **Targets Summary** | Per-target aggregated perturbation stats: expression, cell counts, significant programs, specificity, correlations, KD efficiency |
| **Sample Association** | Kruskal-Wallis + Dunn posthoc p-values per program |
| **Perturbation Association {n}** | Full perturbation results merged with specificity scores (chunked across sheets if >1M rows) |
| **significant regulators only {n}** | Perturbation Association filtered to adj_pval < 0.05, also carrying specificity scores (chunked) |
| **Trait Enrichment** | GWAS trait enrichment via Fisher exact test (Open Targets L2G) |
| **GO Term Enrichment** | GO Biological Process 2023 enrichment |
| **Geneset Enrichment** | Reactome 2022 pathway enrichment |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 64G
- Time: 2h (MyGene API queries for gene annotations are the bottleneck)
