# Stage 3 — Interpretation

After Stage 2 (Evaluation / Calibration) finishes, this stage turns the cNMF runs into the figures, tables, and annotations used to interpret the gene programs. Three independent sub-stages live here; pick whichever you need.

## Pipeline options

| Sub-stage | Purpose | Output | Conda env | Sub-README |
|---|---|---|---|---|
| **A_Plotting** | K-selection panel + per-program report + per-perturbed-gene report | PDF / SVG / HTML | `Interpretation`, `NMF_Benchmarking` | [A_Plotting/README.md](A_Plotting/README.md) |
| **B_Summarization** | Compile Stage 1 + Stage 2 outputs into one multi-sheet `.xlsx` summary | xlsx workbook | `NMF_Benchmarking` | [B_Summarization/README.md](B_Summarization/README.md) |
| **C_Annotation** | LLM-driven gene-program annotation (ProgramExplorer) and literature search | HTML annotation report + supporting CSVs | `progexplorer` | C_Annotation/ProgramExplorer/, C_Annotation/Literature_search/ |

A_Plotting and C_Annotation can run in any order once Stage 2 is done; B_Summarization typically runs last because it pulls outputs from the others.

## Directory layout

```
Stage3_Interpretation/
├── README.md              ← this file
├── environment.yml
├── A_Plotting/
│   ├── README.md          ← full CLI flag reference for the 3 plotting scripts
│   ├── src/               ← plotting library
│   ├── Slurm_Version/     ← cNMF_k_selection / program_analysis / perturbed_gene_analysis (.py + .sh)
│   └── JupterNote_Version/← interactive notebooks for the same 3 tasks
├── B_Summarization/
│   ├── README.md          ← library API + reference notebook workflow
│   ├── src/Compile_excel_sheet.py
│   └── JupterNote_Version/cNMF_compile_excel_table.ipynb
└── C_Annotation/
    ├── ProgramExplorer/   ← run_pipeline.py + numbered steps + Slurm_Version/run_annotation.sh
    └── Literature_search/ ← run_literature_search.py + Slurm_Version/run_literature_search.sh
```

## How to run

Each sub-stage ships SLURM templates in a `Slurm_Version/` folder (and, for A_Plotting and B_Summarization, parallel interactive notebooks in `JupterNote_Version/`). Submit the matching `.sh` from Sherlock or copy the equivalent `python …` command for an interactive run. For per-flag detail, see the sub-READMEs linked in the table above.

## Recommended workflow

1. Run `A_Plotting/cNMF_k_selection.py` first to pick the best K (and density threshold).
2. For the chosen K, run `A_Plotting/cNMF_program_analysis.py` (per-program report) and `A_Plotting/cNMF_perturbed_gene_analysis.py` (per-perturbed-gene report).
3. Run `B_Summarization` to compile everything into a single `.xlsx` workbook.
4. Run `C_Annotation/ProgramExplorer` (and optionally `Literature_search`) to attach LLM-generated annotations to each program.
