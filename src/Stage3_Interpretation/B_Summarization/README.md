# Stage 3b — Excel Summarization

Compile Stage 1 (`.h5mu`) and Stage 2 (Evaluation / Calibration) outputs into a single multi-sheet Excel workbook summarizing every cNMF program and perturbation target. There is no standalone CLI — work is driven by the reference notebook below; the underlying library is reusable in any Python script.

**Conda env**: `NMF_Benchmarking`

**Reference notebook**: `JupterNote_Version/cNMF_compile_excel_table.ipynb`
**Source library**: `src/Compile_excel_sheet.py`

## Directory layout

```
B_Summarization/
├── README.md              ← this file
├── __init__.py
├── src/
│   ├── __init__.py
│   └── Compile_excel_sheet.py   ← all helpers
└── JupterNote_Version/
    └── cNMF_compile_excel_table.ipynb
```

## Inputs

The compile helpers read per-K evaluation outputs from Stage 2 at:

```
{out_dir}/{run_name}/Evaluation/{K}_{thresh}/
├── {K}_GO_term_enrichment.txt
├── {K}_geneset_enrichment.txt
├── {K}_trait_enrichment.txt
├── {K}_perturbation_association_results_{Condition}.txt   (one per condition)
├── {K}_categorical_association_results.txt
└── {K}_Explained_Variance.txt
```

`{thresh}` follows the dot-to-underscore convention used throughout the pipeline (`0.2` → `0_2`, `2.0` → `2_0`).

The Stage 1 `.h5mu` file (typically `{out_dir}/{run_name}/Inference/adata/cNMF_{K}_{thresh}.h5mu`) is loaded by the notebook and passed in as the `mdata` argument.

> If your project uses a non-default evaluation directory name (e.g. `Eval/` instead of `Evaluation/`), **do not** call `load_simple_sheets()` — it hardcodes `Evaluation/`. Build paths manually and call the individual `Compile_*_sheet()` helpers instead.

## API

Exported by `src/__init__.py`:

| Function | Purpose |
|---|---|
| `load_simple_sheets(mdata, out_dir, run_name, k, sel_thresh, ...)` | One-shot loader: reads all per-K evaluation files and returns the loading / GO / geneset / trait / perturbation / association / explained-variance DataFrames. |
| `compile_Program_loading_score_sheet_long(mdata, num_gene)` | Top-N genes per program with `mygene` summary annotations (long form). |
| `compile_Program_loading_score_sheet_flat(mdata, num_gene)` | Top-N genes per program as a flat rank table. |
| `Compile_GO_sheet`, `Compile_Geneset_sheet`, `Compile_Trait_sheet` | Read and truncate the per-K enrichment files (top-N genes per term). |
| `Compile_Perturbation_sheet(base, Sample, sample_key)` | Concatenate per-condition perturbation result files. |
| `Compile_Association_sheet`, `Compile_Explained_variance` | Read the per-K categorical association and explained-variance files. |
| `Compile_Target_Summary_sheet(mdata, perturbation_path, ...)` | Per-target summary: mean expression / cell counts / significant programs / specificity (PMI) / gene-gene correlation / KD efficiency. |
| `Compile_Summary_sheet(mdata, df_GO, df_Geneset, df_Perturbation, df_Program_loading, df_Explained_Variance, ...)` | Per-program summary: GO/geneset top terms, regulator counts, top-loaded genes, mean program score per condition, variance explained, specificity-based top regulators. |
| `add_specificity_scores_file(save_path, perturb_base, samp)` | Merge per-target PMI scores back into a perturbation file. |
| `check_program_name_match(mdata, dataframes, prog_key)` | Sanity-check that program names align across files. |
| `compute_kd_efficiency(mdata, categorical_key, ...)` | Standalone KD efficiency per target per condition (CP10K-normalized). |

## Typical usage

Open the reference notebook, edit the parameter cell, then either run interactively or via `jupyter nbconvert --execute`:

```python
from Stage3_Interpretation.B_Summarization.src import (
    load_simple_sheets, Compile_Target_Summary_sheet, Compile_Summary_sheet,
)
import muon as mu
import pandas as pd

mdata = mu.read_h5mu(f"{out_dir}/{run_name}/Inference/adata/cNMF_{k}_{thresh_str}.h5mu")

(df_loading_long, df_loading_flat, df_GO, df_Geneset, df_Trait,
 df_Perturbation, df_Association, df_ExpVar) = load_simple_sheets(
    mdata, out_dir=out_dir, run_name=run_name,
    k=k, sel_thresh=sel_thresh, num_gene=300,
    Sample=Conditions,                       # e.g. ['D0', 'sample_D1', 'sample_D2', 'sample_D3']
    perturbation_file_name="perturbation_association_results",
)

df_target_summary = Compile_Target_Summary_sheet(
    mdata,
    perturbation_path=f"{out_dir}/{run_name}/Evaluation/{k}_{thresh_str}/{k}_perturbation_association_results",
    Sample=Conditions, save_path=save_path,
)

df_program_summary = Compile_Summary_sheet(
    mdata, df_GO, df_Geneset, df_Perturbation, df_loading_flat, df_ExpVar,
    specicicity_path=save_path, Sample=Conditions,
    non_tagerting_key=['non-targeting'],
)

# Write everything to one .xlsx with one sheet per DataFrame
with pd.ExcelWriter(f"{save_path}/{run_name}_summary.xlsx") as xw:
    df_program_summary.to_excel(xw, sheet_name="Program_summary")
    df_target_summary.to_excel(xw, sheet_name="Target_summary")
    df_loading_flat.to_excel(xw, sheet_name="Program_loadings_flat")
    df_loading_long.to_excel(xw, sheet_name="Program_loadings_long")
    if df_GO is not None: df_GO.to_excel(xw, sheet_name="GO")
    if df_Geneset is not None: df_Geneset.to_excel(xw, sheet_name="Geneset")
    if df_Trait is not None: df_Trait.to_excel(xw, sheet_name="Trait")
    if df_Perturbation is not None: df_Perturbation.to_excel(xw, sheet_name="Perturbation")
    if df_Association is not None: df_Association.to_excel(xw, sheet_name="Categorical_association")
    if df_ExpVar is not None: df_ExpVar.to_excel(xw, sheet_name="Explained_variance")
```

## Outputs

### Directory layout

```
{save_path}/
├── {run_name}_summary.xlsx                       ← main multi-sheet workbook (see Sheet structure below)
├── specificity_score_{Condition}.txt             ← per-condition target × program PMI matrix (one per Condition)
├── corr_gene_matrix_{Condition}.txt              ← per-condition target × target log2FC correlation
├── corr_gene_matrix_{Condition}.txt.gz           ← same, gzipped
└── kd_efficiency.txt                             ← per-target × per-condition KD efficiency (CP10K-normalized)
```

Sidecar files (`specificity_score_*`, `corr_gene_matrix_*`, `kd_efficiency.txt`) are written automatically by `Compile_Target_Summary_sheet(..., save_path=…)` and `compute_kd_efficiency(..., save_path=…)`. They are not produced if `save_path=None`.

### Sheet structure of `{run_name}_summary.xlsx`

| Sheet | One row per | Source helper | Key columns |
|-------|-------------|---------------|-------------|
| **Program_summary** | cNMF program | `Compile_Summary_sheet` | `manual_annotation_label`, `manual_timepoint`, `Notes`, `Automatic Timepoint`, `Total Enriched GO Terms`, `Significant regulators with positive/negative effect {Condition}`, `Top 5 specific regulators (FDR<0.1) {Condition}`, `sigfdr0.05_targets_sorted_abslog2fcd_{Condition}`, `top10_loaded_genes`, `top30_loaded_genes`, `variance_explained`, `Mean program score {Condition}`, `Fra cells above mean program score {Condition}`, `top10_enriched_genesets`, `top10_enriched_go_terms` |
| **Target_summary** | perturbation target | `Compile_Target_Summary_sheet` | `mean_expression_{Condition}`, `# Cells {Condition}`, `significant programs {Condition}`, `# programs {Condition}`, `top 5 specific programs (FDR < 0.1) {Condition}`, `top 5 specificity scores (FDR < 0.1) {Condition}`, `top 5 pos/neg correls targets (program log2fc) {Condition}`, `top 5 pos/neg correls (program log2fc) {Condition}`, `kd_efficiency_{Condition}`, `Mean KD efficiency (%) across conditions` |
| **Program_loadings_flat** | cNMF program | `compile_Program_loading_score_sheet_flat` | Rank `1`..`num_gene` → gene name (top genes per program, descending) |
| **Program_loadings_long** | (program, rank) | `compile_Program_loading_score_sheet_long` | `Program`, `Rank`, `Gene`, `Annotation` (mygene summary) |
| **GO** | enriched GO term × program | `Compile_GO_sheet` | indexed by `Term`; passes through columns from `{K}_GO_term_enrichment.txt`; `Genes` truncated to top `num_gene` |
| **Geneset** | enriched geneset × program | `Compile_Geneset_sheet` | same shape, from `{K}_geneset_enrichment.txt` |
| **Trait** | enriched trait × program | `Compile_Trait_sheet` | same shape, from `{K}_trait_enrichment.txt` |
| **Perturbation** | (target, program, condition) | `Compile_Perturbation_sheet` | `target_name`, `program_name`, `log2FC` (or user-chosen `effect_size`), `adj_pval`, `Sample` |
| **Categorical_association** | cNMF program | `Compile_Association_sheet` | indexed by `program_name`; columns from `{K}_categorical_association_results.txt` |
| **Explained_variance** | cNMF program | `Compile_Explained_variance` | indexed by `program_name`; `VarianceExplained` (and any other variance columns from the file) |

Optional sheets (`GO`, `Geneset`, `Trait`, `Perturbation`, `Categorical_association`, `Explained_variance`) are written only if the corresponding source file exists — `load_simple_sheets()` returns `None` for any file it can't find and the example loop above skips writing those sheets.

## Related

- Stage 3a plotting reports: `../A_Plotting/README.md`
- Stage 3c gene/program annotation pipeline: `../C_Annotation/README.md`
- Skill walkthrough: `.claude/skills/perturbNMF-runner/references/05-annotation-summary.md`
