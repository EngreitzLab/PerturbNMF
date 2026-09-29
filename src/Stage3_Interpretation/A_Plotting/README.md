# Stage 3a — Plotting

After Stage 2 (Evaluation/Calibration) finishes, this stage generates the publication-ready figures and per-program / per-gene reports used to interpret the cNMF runs. Three scripts live here, one per topic:

| Script | What it produces | Reference |
|---|---|---|
| `cNMF_k_selection.py` | K-selection panel (stability vs. error, enrichment counts, perturbation regulators, explained variance) across multiple K and density thresholds | [Example/K-selection_Example.png](Example/K-selection_Example.png) |
| `cNMF_program_analysis.py` | One full report **per program** — UMAP, top genes, top GO terms, perturbation log2FC / volcano / heatmap, correlation waterfall — bundled into a PDF or HTML report | [Example/Program_PDF_Report_Example.pdf](Example/Program_PDF_Report_Example.pdf) |
| `cNMF_perturbed_gene_analysis.py` | One full report **per perturbed gene** — UMAP, top programs the gene affects, perturbation log2FC / volcano, correlation waterfall — bundled into a PDF or HTML report | [Example/Gene_PDF_Report_Example.pdf.pdf](Example/Gene_PDF_Report_Example.pdf.pdf) |

**Conda envs**: `torch-nmf-dl` for K-selection plot; `NMF_Benchmarking` for program & gene reports.

## Directory layout

```
A_Plotting/
├── README.md              ← this file
├── environment.yml
├── Example/               ← sample outputs (PNG + PDF)
├── src/                   ← plotting library
│   ├── k_selection_plots.py
│   ├── k_quality_plots.py             ← programs_dotplots (per-K dot plot)
│   ├── Program_QC_plots.py            ← per-program panels
│   ├── Program_expression_weighted_plots.py
│   ├── Perturbed_gene_QC_plots.py     ← per-gene panels
│   ├── html_Program_QC_plots.py       ← HTML report assembly (programs)
│   ├── html_Perturbed_gene_QC_plots.py← HTML report assembly (genes)
│   └── utilities.py
├── Slurm_Version/         ← SLURM-ready scripts + sample .sh templates
└── JupterNote_Version/    ← interactive notebooks for the same 3 tasks
```

## 1. K-Selection Plot

Generates a multi-panel figure to guide K selection. Reads pre-computed Stage 2 evaluation outputs from `<eval_folder_name>/{K}_{thresh}/`.

**Script**: `Slurm_Version/cNMF_k_selection.py` (sibling `cNMF_k_selection.sh`)
**Notebook**: `JupterNote_Version/cNMF_k_selection.ipynb`

### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output_directory` | str | **required** | Inference output root (`{OUT_DIR}/{RUN_NAME}/`) |
| `--run_name` | str | **required** | cNMF run name |
| `--groupby` | str | `sample` | Column for grouping in analysis |
| `--K` | int list | `30 50 70 80 100 200 300` | K values (number of components) to plot |
| `--save_folder_name` | str | **required** | Output directory for plots |
| `--pval` | float | `0.05` | P-value threshold |
| `--eval_folder_name` | str | **required** | Path to Stage 2 Evaluation results |
| `--sel_threshs` | float list | `0.2 2.0` | Density thresholds |
| `--Conditions` | str list | all labels found in the data | Condition labels (values of obs[categorical_key]), e.g. `condA condB` |
| `--selected_k` | int | `None` | K value to highlight with a red dashed line |
| `--go_file` | str | `{k}_GO_term_enrichment.txt` | GO enrichment file name pattern (use `{k}` for the K value) |
| `--geneset_file` | str | `{k}_geneset_enrichment.txt` | Geneset enrichment file name pattern |
| `--trait_file` | str | `{k}_trait_enrichment.txt` | Trait enrichment file name pattern |
| `--term_col` | str | `Term` | Column name for the term/pathway name in enrichment files |
| `--adjpval_col` | str | `Adjusted P-value` | Column name for adjusted p-value in enrichment files |
| `--perturbation_file` | str | `{k}_perturbation_association_results_{sample}.txt` | Perturbation file name pattern (use `{k}` and `{sample}` placeholders) |
| `--perturb_adjpval_col` | str | `adj_pval` | Column name for adjusted p-value in perturbation files |
| `--perturb_target_col` | str | `target_name` | Column name for target/regulator name in perturbation files |
| `--perturb_log2fc_col` | str | `log2FC` | Column name for log2 fold change in perturbation files |
| `--variance_file` | str | `{k}_Explained_Variance_Summary.txt` | Explained variance file name pattern |
| `--variance_col` | str | `Total` | Column name for variance values. Use `Total` for summary files, or a per-program column name (will be summed) |
| `--stability_file` | str | `None` | Path to a pre-computed stability/error file (TSV or NPZ). Bypasses `cnmf.consensus()` — useful for torch-cNMF runs where the cnmf package isn't installed |
| `--run_program_dotplot` | flag | off | If set, also generate per-(K, sel_thresh) program dotplots. Requires the inference `cNMF_{K}_{thresh}.h5mu` to exist for each pair; off by default because the loop is slow |

### Outputs

- `Stability_Error_stability.pdf/svg/png` and `Stability_Error_error.pdf/svg/png` — stability + reconstruction error curves
- `Enrichment_{thresh}_{go_terms,genesets,traits}.pdf/svg/png` — enrichment counts vs. K
- `Perturbation_{thresh}_per_condition.pdf/svg/png` — unique regulators per condition (one line per condition)
- `Perturbation_{thresh}_all_conditions.pdf/svg/png` — unique regulators aggregated across all conditions
- `Explained_Variance_{thresh}.pdf/svg/png`
- `K-selection_panel_{thresh}.pdf/svg/png` — combined panel (the "official" K-selection figure)
- (optional, with `--run_program_dotplot`) `Program_dotplot_{K}_{thresh}.png` per K × threshold

## 2. Program Analysis Report (PDF + HTML)

Produces one comprehensive panel per cNMF program. Loops over all programs in the supplied `.h5mu` and assembles them into a single PDF (and optionally a shareable HTML page).

**Script**: `Slurm_Version/cNMF_program_analysis.py` (sibling `cNMF_program_analysis.sh`)
**Notebook**: `JupterNote_Version/cNMF_program_analysis.ipynb`

### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--mdata_path` | str | **required** | Path to `cNMF_{K}_{thresh}.h5mu` (Stage 1 output) |
| `--perturb_path_base` | str | `None` | Base path for per-sample perturbation result files (sample suffix appended). Omit to skip every per-condition perturbation panel and the regulator heatmap — see "Running without perturbation results" below. Required with `--output_format HTML` |
| `--file_to_dictionary` | str | `None` | Path to gene name mapping dictionary file for Ensembl-ID-to-symbol conversion |
| `--reference_gtf_path` | str | `None` | Path to reference GTF file for checking gene names |
| `--GO_path` | str | **required** | Path to GO enrichment results directory |
| `--tagert_col_name` | str | `program_name` | Column name for target programs in perturbation results (sic — preserved spelling matches the flag) |
| `--plot_col_name` | str | `target_name` | Column name for genes in perturbation results |
| `--log2fc_col` | str | `log2FC` | Column name for log2 fold change values |
| `--top_program` | int | `10` | Number of top programs to display |
| `--top_enrichned_term` | int | `10` | Number of top GO enrichment terms to display per program (sic — preserved spelling matches the flag) |
| `--p_value` | float | `0.05` | P-value threshold for significance |
| `--down_thred_log` | float | `-0.00` | Lower log2FC threshold for volcano plot |
| `--up_thred_log` | float | `0.00` | Upper log2FC threshold for volcano plot |
| `--save_path` | str | **required** | Directory path to save output (PDF/SVG files or HTML share tree) |
| `--square_plots` | flag | off | Auto-scale figure height to the number of conditions (`num_rows × 8 in`) so panels stay roughly square; only the width from `--figsize` is used |
| `--figsize` | float list (2) | `35 35` | Figure size as `width height`. With `--square_plots`, only width is used (height auto-scales with condition count) |
| `--show` | flag | off | Display plots interactively |
| `--output_format` | str (choice) | `SVG` | One of `PDF` / `SVG` / `HTML`. `HTML` writes per-program interactive Plotly pages directly under `save_path` |
| `--Conditions` | str list | all labels found in the data | Condition labels (values of obs[categorical_key]), e.g. `condA condB` |
| `--programs` | int list | `None` (all) | Specific program numbers to plot (e.g. `4 5 6`). If omitted, every program in the h5mu is plotted |
| `--subsample_frac` | float | `None` (all) | Fraction of cells to subsample for UMAP plots (e.g. `0.1` for 10%) |
| `--corr_matrix_path` | str | `None` | Directory for the full program × program correlation matrices written every run: `<dir>/program_corr.npz` (usage correlation) and `<dir>/program_waterfall_corr_<sample>.npz` (perturbation-effect correlation). Defaults to `--save_path`. See [Full correlation matrices](#full-correlation-matrices) |
| `--skip_existing` | flag | on (default) | Default behavior **skips** programs whose output already exists. Passing `--skip_existing` turns OFF skipping and re-processes every program from scratch (handy for resuming preempted jobs) |
| `--data_key` | str | `rna` | Key to access gene expression data in MuData |
| `--prog_key` | str | `cNMF` | Key to access cNMF programs in MuData |
| `--gene_name_key` | str | `gene_names` | Key to access gene names in var |
| `--categorical_key` | str | `sample` | Key in .obs holding each cell's condition label (e.g. timepoint, stimulus, donor) |

### Outputs per program

- UMAP colored by program usage
- Top loaded genes (bar + dot)
- Top GO terms (bar)
- Perturbation log2FC scatter / volcano / heatmap
- Program-program correlation waterfall
- Combined panel → one page in the PDF

Set `--output_format HTML` to emit a shareable HTML report under `save_path` (per-program subdirectories).

### Running without perturbation results

`--perturb_path_base` is optional. When it is omitted, the report drops everything derived from the per-condition association files and emits a single-row panel per program built from the `.h5mu` plus the GO enrichment table:

- **Row 0** — UMAP program usage | program expression violin | top loading genes | GO enrichment | program–program loading correlation

The per-condition rows (log2FC, volcano, regulator dotplot, waterfall) and the regulator-effect heatmap row are skipped, as is the waterfall correlation precompute. `--GO_path` is still required. This lets you QC program structure and enrichment before Stage 2b calibration (CRT / U-test) has been run. `--output_format HTML` still requires `--perturb_path_base`; use `PDF` or `SVG` in this mode.

Sample output: [Example/Program_PDF_Report_Example.pdf](Example/Program_PDF_Report_Example.pdf).

## 3. Perturbed Gene Analysis Report (PDF + HTML)

Produces one comprehensive panel per **perturbed gene** — counterpart to the program report, but indexed by gene rather than by program. For each gene, shows which programs it affects and how.

**Script**: `Slurm_Version/cNMF_perturbed_gene_analysis.py` (sibling `cNMF_perturbed_gene_analysis.sh`)
**Notebook**: `JupterNote_Version/cNMF_perturbed_gene_analysis.ipynb`

### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--mdata_path` | str | **required** | Path to `cNMF_{K}_{thresh}.h5mu` (Stage 1 output) |
| `--perturb_path_base` | str | `None` | Base path for per-sample perturbation result files (sample suffix appended). Omit to skip every per-condition perturbation panel — see "Running without perturbation results" below. Required with `--output_format HTML` |
| `--ensembl_to_symbol_file` | str | `None` | Path to gene name mapping dictionary file for Ensembl-ID-to-symbol conversion |
| `--reference_gtf_path` | str | `None` | Path to reference GTF file for checking gene names |
| `--perturb_target_col` | str | `target_name` | Column name for target genes in perturbation results |
| `--perturb_program_col` | str | `program_name` | Column name for programs in perturbation results |
| `--perturb_log2fc_col` | str | `log2FC` | Column name for log2 fold change values |
| `--top_corr_genes` | int | `5` | Number of top correlated genes to display per program |
| `--top_n_programs` | int | `10` | Number of top programs to display per gene |
| `--significance_threshold` | float | `0.05` | P-value threshold for significance |
| `--volcano_log2fc_min` | float | `-0.00` | Lower log2FC threshold for volcano plot |
| `--volcano_log2fc_max` | float | `0.00` | Upper log2FC threshold for volcano plot |
| `--save_path` | str | **required** | Directory path to save output (PDF/SVG files or HTML share tree) |
| `--square_plots` | flag | off | Auto-scale figure height to the number of conditions (`num_rows × 8 in`) so panels stay roughly square; only the width from `--figsize` is used |
| `--figsize` | float list (2) | `35 35` | Figure size as `width height`. With `--square_plots`, only width is used (height auto-scales with condition count) |
| `--show` | flag | off | Display plots interactively |
| `--output_format` | str (choice) | `SVG` | One of `PDF` / `SVG` / `HTML`. `HTML` writes per-gene interactive Plotly pages directly under `save_path` |
| `--n_processes` | int | `-1` | Number of parallel processes (`-1` = all available cores) |
| `--Conditions` | str list | all labels found in the data | Condition labels (values of obs[categorical_key]), e.g. `condA condB` |
| `--umap_dot_size` | int | `10` | Dot size for UMAP plots |
| `--expressed_only` | flag | off | Only plot perturbed genes found in the gene expression matrix (default plots all perturbed genes) |
| `--gene_list_file` | str | `None` | Path to a file with one gene name per line to process (overrides automatic perturbed gene detection) |
| `--subsample_frac` | float | `None` (all) | Fraction of cells to subsample for UMAP plots (e.g. `0.1` for 10%) |
| `--parallel` | flag | off | Use fork-based multiprocessing to plot genes in parallel (Linux only) |
| `--corr_matrix_path` | str | `None` | Directory for the full correlation matrices written every run: regulator × regulator `<dir>/regulator_corr_<sample>.npz` and gene × gene `<dir>/gene_loading_corr.npz`. Defaults to `--save_path` |
| `--no_save_regulator_corr` | flag | off | Do not write `regulator_corr_<sample>.npz` (~1.6 GB per sample at 20k regulators). Plots are unaffected |
| `--no_save_gene_corr` | flag | off | Do not write `gene_loading_corr.npz` (~0.4 GB at 10k genes, ~3.6 GB at 30k). Plots are unaffected |
| `--skip_existing` | flag | on (default) | Default behavior **skips** genes whose output already exists. Passing `--skip_existing` turns OFF skipping and re-processes every gene from scratch |
| `--data_key` | str | `rna` | Key to access gene expression data in MuData |
| `--prog_key` | str | `cNMF` | Key to access cNMF programs in MuData |
| `--gene_name_key` | str | `gene_names` | Key to access gene names in var |
| `--categorical_key` | str | `sample` | Key in .obs holding each cell's condition label (e.g. timepoint, stimulus, donor) |
| `--guide_targets_key` | str | `guide_targets` | Key in `.uns` to access guide target genes |
| `--control_target_name` | str (nargs='+') | `non-targeting` | One or more control labels in `guide_targets` (e.g. `non-targeting`, or `WT WT111 WT4`). A cell is a control if its guide target matches **any** of these. Use multiple labels when controls are background-specific. These targets are excluded from the per-gene perturbation panels (no association results to plot) |

### Outputs per gene

- UMAP colored by guide assignment for this gene
- Top N programs the gene perturbs
- Perturbation log2FC scatter + volcano
- Gene-program correlation waterfall
- Per-perturbation vs control comparison
- Combined panel → one page in the PDF (or HTML page)

Supports parallel processing via `--n_processes` and `--parallel` (Linux fork-based multiprocessing). Use `--gene_list_file` to restrict to a subset of genes, or `--expressed_only` to skip perturbed-but-unexpressed genes.

### Full correlation matrices

Both reports write their full correlation matrices to `--corr_matrix_path` every run as uncompressed `.npz` (code: `src/efficient_corr.py`). The gene and regulator matrices can be 20k+ × 20k+, so plots compute one row at a time (memory N × K); only saving builds the full matrix, once, in RAM (N² × 4 bytes: 1.6 GB at 20k, 3.6 GB at 30k; skip with `--no_save_regulator_corr` / `--no_save_gene_corr`). The program matrices are only K × K; the usage correlation is streamed over cells instead of copying the cells × K matrix.

| Report | File | Rows / columns | Correlates | Names key |
|---|---|---|---|---|
| perturbed gene | `regulator_corr_<sample>.npz` (one per condition) | perturbed regulators | per-program log2FC profiles (waterfall panel) | `regulator` |
| perturbed gene | `gene_loading_corr.npz` | genes | cNMF loading profiles (gene-loading correlation panel) | `gene` |
| program | `program_corr.npz` | programs | cNMF usages across cells (program correlation panel) | `program` |
| program | `program_waterfall_corr_<sample>.npz` (one per condition) | programs | log2FC profiles across regulators (waterfall panel) | `program` |

Each file is a plain `np.savez` with `corr` (float32 Pearson; NaN for an item with a constant profile) and the names key (plain strings, so no `allow_pickle` needed). The regulator × regulator matrix needs a log2FC for every regulator × program pair; a missing value raises an error. The two program matrices are only K × K: the waterfall one is plain pandas `.corr()`. The gene rows used by the plots are returned as computed (NaN kept); only the program-usage panel fills NaN with 0 (legacy `.fillna(0)`); the saved files keep NaN.

```python
import numpy as np
with np.load("regulator_corr_D0.npz") as f:
    names = list(f["regulator"])
    sox2 = f["corr"][names.index("SOX2")]      # loads the full matrix, then takes one row
```

### Running without perturbation results

`--perturb_path_base` is optional. When it is omitted, the report drops everything derived from the per-condition association files and emits a two-row panel per gene built entirely from the `.h5mu`:

- **Row 0** — UMAP expression | UMAP perturbation | gene expression dotplot | top loading programs
- **Row 1** — CRISPRi knockdown grouped bar (All + per condition) | gene-loading correlation

The per-condition rows (log2FC, volcano, program dotplot, perturbation waterfall) and the waterfall correlation precompute are skipped. This lets you QC guide assignment and knockdown efficiency before Stage 2b calibration (CRT / U-test) has been run. `--output_format HTML` still requires `--perturb_path_base`; use `PDF` or `SVG` in this mode.

Sample output: [Example/Gene_PDF_Report_Example.pdf.pdf](Example/Gene_PDF_Report_Example.pdf.pdf).

## Common conventions

- **Output format**: each script supports `--output_format {PDF, SVG, HTML}`. PDF is default for sharing; HTML mode emits a self-contained report viewable in any browser, written directly under `--save_path`.
- **Figure size**: `--figsize WIDTH HEIGHT` (default `35 35`). Used verbatim unless `--square_plots` is set.
- **Square aspect**: `--square_plots` auto-scales the figure height to the number of conditions (`num_rows × 8 in`) so panels stay roughly square (only the width from `--figsize` is used). Applies to both the **Program Analysis** (section 2) and **Perturbed Gene Analysis** (section 3) reports — recommended for runs with many conditions, which otherwise get squished into a fixed square.
- **Subsampling**: `--subsample_frac 0.1` to render UMAP from 10% of cells (useful for >500k-cell runs).
- **Custom column names**: each script accepts overrides for the `target_col`, `program_col`, `log2fc_col`, `adjpval_col` so it works on non-default perturbation result schemas (e.g. Morphic data).

## What's next

After this stage produces reports, proceed to:

- **Stage 3b — Excel Summarization** (`../B_Summarization/`) — compile a single `.xlsx` summary across all programs / genes.
- **Stage 3c — LLM Annotation** (`../C_Annotation/`) — auto-annotate program identities using the per-program data exported here.
