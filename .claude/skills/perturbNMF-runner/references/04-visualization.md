# Visualization Stages (K-Selection / Program Analysis / Perturbed Gene)

> **All-flags convention (mandatory for every generated `.sh`):** When invoking `generate_slurm.py`, list active flags first, then `---COMMENTED---`, then every remaining flag for this stage from `references/parameter-catalog.md` (Sections 4–6 — K-Selection, Program Analysis, Perturbed Gene) with a sensible default/example value. The generator emits unused flags as `#     --flag value` lines below the python command so the user can toggle them later. See `SKILL.md` Step 5.

---

## K-Selection Plot (Stage 3a)

**Conda**: `torch-nmf-dl`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--output_directory` | Directory with cNMF output |
| `--run_name` | cNMF run name |
| `--save_folder_name` | Where to save plots |
| `--eval_folder_name` | Path to Eval results |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--K` | `[30, 50, 70, 80, 100, 200, 300]` | K values |
| `--sel_threshs` | `[0.2, 2.0]` | Density thresholds |
| `--Conditions` | all unique values of the categorical key in the h5mu | Condition labels matching the categorical key used during evaluation |
| `--groupby` | `sample` | Grouping variable |
| `--pval` | `0.05` | P-value threshold |
| `--selected_k` | None | K value to highlight with a red dashed line |
| `--stability_file` | None | Pre-computed stability/error TSV/NPZ (bypasses `cnmf.consensus()`; for torch-cNMF runs) |
| `--run_program_dotplot` | off | Enable per-(K, sel_thresh) program dotplots (default off; requires the inference h5mu artifact per K/threshold) |

Enrichment file patterns use `{k}` placeholder (and `{sample}` for perturbation). See `references/parameter-catalog.md` Section 4 for all optional params.

### SLURM resources

- Partition: standard CPU partition (ask the user), CPUs: 4, Memory: 32-64G, Time: 1h

---

## Program Analysis Plot (Stage 3b)

**Conda**: `NMF_Benchmarking`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--mdata_path` | Path to .h5mu file |
| `--GO_path` | Path to GO enrichment results |
| `--save_path` | Output directory (PDF/SVG files or HTML share tree) |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--perturb_path_base` | None | Base path for perturbation results. Omit to skip every per-condition perturbation panel (log2FC, volcano, regulator dotplot, waterfall) and the regulator heatmap, emitting only the h5mu/GO-derived header row — useful before Stage 2b calibration has been run. Required with `--output_format HTML` |
| `--top_program` | `10` | Top programs to display |
| `--top_enrichned_term` | `10` | Top GO terms per program (**note typo**: use as-is) |
| `--p_value` | `0.05` | Significance threshold |
| `--programs` | None | Specific program numbers (e.g. `4 5 6`); if omitted, all plotted |
| `--subsample_frac` | None | Fraction of cells to subsample for UMAP |
| `--output_format` | `SVG` | One of `PDF` / `SVG` / `HTML` |
| `--corr_matrix_path` | None | Directory for the full program × program matrices written every run: `program_corr.npz` and `program_waterfall_corr_<sample>.npz`; defaults to `--save_path` |
| `--skip_existing` | on (default) | Default skips programs whose output already exists. Pass `--skip_existing` to force re-process all (inverted flag) |
| `--tagert_col_name` | `program_name` | Column name for target programs in perturbation results (**note typo**: use as-is) |

See `references/parameter-catalog.md` Section 5 for all optional params.

### SLURM resources

- Partition: standard CPU partition (ask the user), CPUs: 4-8, Memory: 64-128G, Time: 2-4h

---

## Perturbed Gene Plot (Stage 3c)

**Conda**: `NMF_Benchmarking`

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--mdata_path` | Path to .h5mu file |
| `--save_path` | Output directory for plots |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--perturb_path_base` | None | Base path for perturbation results. Omit to skip every per-condition perturbation panel (log2FC, volcano, program dotplot, waterfall) and emit only the two h5mu-derived rows — useful before Stage 2b calibration has been run. Required with `--output_format HTML` |
| `--top_n_programs` | `10` | Top programs to display per gene |
| `--top_corr_genes` | `5` | Top correlated genes per program |
| `--significance_threshold` | `0.05` | P-value threshold |
| `--gene_list_file` | None | File with gene names to process (one per line) |
| `--subsample_frac` | None | Fraction of cells to subsample for UMAP |
| `--parallel` | off | Use fork-based multiprocessing (Linux only) |
| `--n_processes` | `-1` | Number of parallel processes (`-1` = all available cores) |
| `--expressed_only` | off | Only plot perturbed genes found in the gene expression matrix |
| `--guide_targets_key` | `guide_targets` | Key in `.uns` to access guide target genes |
| `--guide_assignment_key` | `guide_assignment` | Key in `.obsm` to access the guide-assignment matrix |
| `--control_target_name` | `non-targeting` | One or more control labels in `guide_targets` (e.g. `non-targeting`, or `WT WT111 WT4`); a cell is a control if it matches any. Use multiple labels for background-specific controls. Excluded from the per-gene perturbation panels (no association results to plot) |
| `--corr_matrix_path` | None | Directory for the full matrices written every run: regulator × regulator `regulator_corr_<sample>.npz` and gene × gene `gene_loading_corr.npz`; defaults to `--save_path` |
| `--no_save_regulator_corr` | off | Skip writing `regulator_corr_<sample>.npz` (~1.6 GB per sample at 20k regulators); plots unaffected |
| `--no_save_gene_corr` | off | Skip writing `gene_loading_corr.npz` (~3.6 GB at 30k genes); plots unaffected |
| `--skip_existing` | on (default) | Default skips genes whose output already exists. Pass `--skip_existing` to force re-process all (inverted flag) |
| `--output_format` | `SVG` | One of `PDF` / `SVG` / `HTML` |

See `references/parameter-catalog.md` Section 6 for all optional params.

### SLURM resources

- Partition: standard CPU partition (ask the user), CPUs: 4-10, Memory: 64-128G, Time: 2-6h
