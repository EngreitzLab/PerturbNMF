# Evaluation Stage

> **All-flags convention (mandatory for every generated `.sh`):** When invoking `generate_slurm.py`, list active flags first, then `---COMMENTED---`, then every remaining flag for this stage from `references/parameter-catalog.md` (Section 3 — Evaluation) with a sensible default/example value. The generator emits unused flags as `#     --flag value` lines below the python command so the user can toggle them later. See `SKILL.md` Step 5.

## Step A: Read inference config to auto-populate parameters

```bash
cat <out_dir>/<run_name>/Inference/config_*.yml
```

Extract: `K`, `sel_thresh`, `categorical_key`, `gene_names_key`, `species` -> `organism`, `data_key`, `prog_key`.

## Step B: Construct paths

- `--out_dir`: Parent directory containing the run (e.g., `Result/`)
- `--run_name`: The run directory name
- `--X_normalized_path`: `<out_dir>/<run_name>/Inference/cnmf_tmp/Inference.norm_counts.h5ad`

## Step C: Determine which tests to run (9 metrics total)

| Flag | Description | Notes |
|------|-------------|-------|
| `--Perform_categorical` | Categorical association (Kruskal-Wallis + Dunn's) | |
| `--Perform_perturbation` | Perturbation sensitivity | **Requires guide data; skip for bulk RNA-seq** |
| `--Perform_motif` | TF motif enrichment + candidate TFs | Needs a MEME file, genome FASTA and GTF (flags or env vars, below) |
| `--Perform_trait` | GWAS trait enrichment | Requires `--gwas_data_path` |
| `--Perform_geneset` | GO + geneset enrichment (Reactome, MsigDB) | |
| `--Perform_explained_variance` | Explained variance per K | Needs `--X_normalized_path` |

Reconstruction error and stability are computed automatically.

## Step D: Optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gwas_data_path` | None | Path to GWAS data (use GWAS_DATA constant) |
| `--guide_annotation_path` | None | TSV with a `targeting` column to identify non-targeting controls (alternative to `--guide_annotation_key`) |
| `--gene_names_key` | `symbol` | Column in data_guide["rna"].var with gene names |
| `--FDR_method` | `StoreyQ` | FDR correction: `StoreyQ` or `BH` |
| `--organism` | `human` | Species for enrichment |
| `--n_top` | `300` | Top genes for enrichment tests |
| `--guide_annotation_key` | `["non-targeting"]` | Non-targeting guide identifiers (accepts multiple values) |
| `--use_cache` | flag | Load enrichr gene set libraries from cached JSON in `Resources/` instead of downloading; falls back to download + cache on miss |
| `--skip_existing` | flag | Skip metric computations whose output files already exist on disk; useful for resuming preempted batches |
| `--reassign_name` | flag | Reassign `mdata[data_key].var_names` from `var[gene_names_key]` before running metrics (use when var index is Ensembl IDs) |

### TF motif enrichment (`--Perform_motif`)

Same options in `cNMF_evaluation_pipeline.py` and the standalone driver
`src/Stage2_Evaluation/A_Metrics/Slurm_Version/run_motif_enrichment.py` (+ `.sh`). Writes
`{K}_motif_enrichment.txt`, `{K}_candidate_tfs.txt`, `{K}_motif_logos.json` per K. Full table:
`src/Stage2_Evaluation/A_Metrics/README.md#motif-enrichment`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--motif_method` | `ttest` | `ttest` (top `--n_top` genes vs background, Welch) or `correlation` (`--motif_correlation pearson\|spearman`) |
| `--motif_source` | `fimo` | `fimo`, `finemo` (ChromBPNet Fi-NeMo hits) or `both` (never pooled) |
| `--motif_element_types` | `promoter enhancer` | element types to test |
| `--motif_db` | `motifcompendium` | FIMO database: `motifcompendium`, `hocomoco_v11`, or a MEME path |
| `--motif_file` | `$PERTURBNMF_MOTIFCOMPENDIUM_MEME` | MEME file to scan (`$PERTURBNMF_HOCOMOCO_V11_MEME` with `hocomoco_v11`) |
| `--genome_fasta` | `$PERTURBNMF_GENOME_FASTA` | genome FASTA (build = `--genome_build`, default hg38) |
| `--gene_annotation` | `$PERTURBNMF_GENE_ANNOTATION` | GTF or BED6 for promoter windows |
| `--enhancer_links` | None | element-gene links (ABC / ENCODE-rE2G / scE2G tsv, IGVF bedpe); or `--regulatory_resources_manifest` from `find_regulatory_resources.py` |
| `--finemo_instances` / `--finemo_report` | None | Fi-NeMo hits + report (ENCODE tars or local tables with `--finemo_annotation`) |
| `--fimo_binary` / `--n_jobs` | `fimo` / 1 | MEME fimo and parallel chunks |
| `--motif_fdr_threshold` | `0.05` | significance and candidate-TF FDR |
| `--motif_hit_cache_dir` | `Evaluation/motif_hits` | hit tables shared by all K |

## SLURM Resources

- Partition: `engreitz,owners,bigmem`
- CPUs: 10-20
- Memory: 64-256G
- Time: 3-5h
