# Explorer

Interactive reports for the datasets in the PerturbNMF paper. Each program page shows its top
genes, GO enrichment, the perturbations that regulate it, and how it varies across conditions.
The pages are self-contained Plotly HTML, so nothing needs to be installed.

::::{grid} 1
:::{grid-item-card} Open the program reports →
:link: https://ymo6.github.io/program-reports/

Pick a dataset and K, then browse the programs.
:::
::::

## Build reports for your own data

The same reports come out of Stage 3 when you set `--output_format HTML`:

```bash
python3 src/Stage3_Interpretation/A_Plotting/Slurm_Version/cNMF_program_analysis.py \
    ... --perturb_path_base <path> --output_format HTML
```

HTML output needs `--perturb_path_base`. The output folder is portable: copy it to any static host, such as GitHub Pages.
See {doc}`guide/plotting`.
