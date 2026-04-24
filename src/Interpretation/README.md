# cNMF Interpretation Pipeline

Comprehensive visualization tools for cNMF (consensus Non-negative Matrix Factorization) analysis results.

## Overview

This module provides three main plotting pipelines for different aspects of cNMF analysis:

1. **K-Selection Analysis** - Optimal K value selection and model comparison
2. **Perturbed Gene Analysis** - Individual gene perturbation effects visualization  
3. **Program Analysis** - cNMF program characterization and annotation

## Components

### 1. K-Selection (`cNMF_k_selection.py`)

Generates plots to help select optimal K values and compare model performance.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| output_directory | str | - | Directory containing cNMF output files |
| run_name | str | - | Name of the cNMF run |
| save_folder_name | str | - | Output directory for plots |
| eval_folder_name | str | - | Directory containing evaluation results |
| groupby | str | "sample" | Column for grouping in analysis |
| K | list of int | [30, 50, 60, 80, 100, 200, 250, 300] | K values to analyze |
| sel_threshs | list of float | [0.4, 0.8, 2.0] | Selection thresholds |
| samples | list of str | ['D0', 'sample_D1', 'sample_D2', 'sample_D3'] | Sample names |
| pval | float | 0.05 | P-value threshold |

#### Outputs
- Stability & error plots across K values
- Program enrichment plots for each threshold
- Perturbation effect analysis
- Explained variance plots
- Program dotplots for detailed visualization

### 2. Perturbed Gene Analysis (`cNMF_perturbed_gene_analysis.py`)

Creates comprehensive analysis plots for each perturbed gene.

#### Parameters

**I/O Paths:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mdata_path | str | - | Path to multimodal data file (.h5mu) |
| perturb_path_base | str | - | Base path for perturbation data files |
| pdf_save_path | str | - | Output path for PDF files |
| file_to_dictionary | str | None | Path to gene name mapping dictionary file for ID-to-name conversion |
| reference_gtf_path | str | None | Path to reference GTF file for checking gene names |

**Visualization Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tagert_col_name | str | "target_name" | Column name for target genes in perturbation results |
| plot_col_name | str | "program_name" | Column name for programs in perturbation results |
| log2fc_col | str | "log2FC" | Log2 fold change column |
| top_num | int | 5 | Number of top genes to display per program |
| top_program | int | 10 | Number of top programs to display per gene |
| p_value | float | 0.05 | P-value threshold for significance |
| down_thred_log | float | -0.05 | Lower log2FC threshold for volcano plot |
| up_thred_log | float | 0.05 | Upper log2FC threshold for volcano plot |
| figsize | list | [35, 35] | Figure size as width height |
| square_plots | flag | False | Use square aspect ratio for plots |
| show | flag | False | Display plots interactively |
| PDF | flag | False | Save as PDF (default is SVG) |
| n_processes | int | -1 | Number of parallel processes (-1 for all available cores) |
| sample | list of str | None | List of sample names to analyze |
| dot_size | int | 10 | Dot size for UMAP plots |
| expressed_only | flag | False | Only plot perturbed genes found in the gene expression matrix |
| subsample_frac | float | None | Fraction of cells to subsample for UMAP plots (e.g. 0.1 for 10%) |

#### Outputs
- Individual gene analysis PDFs with UMAP, correlation plots, volcano plots
- Gene correlation matrices
- Waterfall correlation plots
- Merged comprehensive PDF reports

### 3. Program Analysis (`cNMF_program_analysis.py`)

Generates detailed characterization plots for each cNMF program.

#### Parameters

**I/O Paths:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mdata_path | str | - | Path to multimodal data file (.h5mu) |
| perturb_path_base | str | - | Base path for perturbation data files |
| GO_path | str | - | Path to Gene Ontology enrichment results directory |
| pdf_save_path | str | - | Output path for PDF files |
| file_to_dictionary | str | None | Path to gene name mapping dictionary file for ID-to-name conversion |
| reference_gtf_path | str | None | Path to reference GTF file for checking gene names |

**Analysis Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tagert_col_name | str | "program_name" | Column name for target programs in perturbation results |
| plot_col_name | str | "target_name" | Column name for genes in perturbation results |
| log2fc_col | str | "log2FC" | Log2 fold change column |
| top_program | int | 10 | Number of top programs to display |
| top_enrichned_term | int | 10 | Number of top GO enrichment terms per program |
| p_value | float | 0.05 | P-value threshold for significance |
| down_thred_log | float | -0.05 | Lower log2FC threshold for volcano plot |
| up_thred_log | float | 0.05 | Upper log2FC threshold for volcano plot |
| figsize | list | [35, 35] | Figure size as width height |
| square_plots | flag | False | Use square aspect ratio for plots |
| show | flag | False | Display plots interactively |
| PDF | flag | False | Save as PDF (default is SVG) |
| sample | list of str | None | List of sample names to analyze |
| programs | list of int | None | Specific program numbers to plot (if omitted, all programs are plotted) |
| subsample_frac | float | None | Fraction of cells to subsample for UMAP plots |

#### Outputs
- Program-specific UMAP visualizations
- Top genes per program analysis
- Gene Ontology enrichment plots
- Program correlation matrices
- Violin plots and heatmaps
- Comprehensive PDF reports

## Common Data Keys

All plotting scripts use these configurable keys for data access:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| data_key | str | "rna" | Key to access gene expression in MuData |
| prog_key | str | "cNMF" | Key to access cNMF programs in MuData |
| gene_name_key | str | "gene_names" | Key to access gene names in var |
| categorical_key | str | "sample" | Key to access sample/condition labels in obs |

 

## Output Organization

```
save_folder/
├── K_selection_plots/
│   ├── stability_plots.png
│   ├── enrichment_plots/
│   └── variance_plots/
├── gene_analysis/
│   ├── individual_genes/
│   └── merged_reports.pdf
└── program_analysis/
    ├── individual_programs/
    └── merged_reports.pdf
```

## 4. Annotation (`Annotation/`)

LLM-driven gene program annotation pipeline (ported from ProgExplorer). Uses STRING enrichment, PubTator3 literature mining, and Claude AI to generate structured biological annotations for each gene program.

### Usage

```bash
# Full pipeline via SLURM
sbatch Annotation/Slurm_Version/run_annotation.sh --config Annotation/configs/pipeline_config.yaml

# Or directly
python Annotation/src/run_pipeline.py --config Annotation/configs/pipeline_config.yaml
```

### Preprocessing utilities

```bash
# Convert cNMF gene_spectra_score TSV to long-format CSV
python Annotation/src/convert_spectra_to_csv.py input.txt -o output.csv

# Merge per-timepoint regulator results
python Annotation/src/compile_regulators.py /path/to/data --days D0 D1 D2 D3
```