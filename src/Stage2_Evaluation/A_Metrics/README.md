

# cNMF Evaluation Pipeline

Comprehensive evaluation metrics for cNMF program quality assessment across multiple biological and technical dimensions.

## Overview

The evaluation pipeline tests cNMF programs against various criteria to assess their biological relevance and technical quality. Each program is systematically evaluated using statistical tests and enrichment analyses.

## Evaluation Criteria

| Criterion    | Implementation | External resource | Interpretation | Caveats |
| -------- | ------- | -------- | ------- | ------- |
| Categorical association | [Kruskal-Wallis non-parametric ANOVA](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance) + Dunn's posthoc test | None | If program scores are variable between batch levels then the component likely is modelling technical noise. Alternatively, if program scores are variable between a biological category like cell-type or condition then the program is likely modelling a biological process specific to the category. | If batches are confounded with biological conditions, then the relative contribution of technical and biological variation cannot be decomposed. |
| Perturbation sensitivity | Mann-Whitney U test of program scores between perturbed cells and non-targeted/reference cells | Perturbation data | Cell × program score distribution shifts greater than expected due to the direct effect of perturbation on genes in the program could indicate hierarchical relationships between genes in the program. | Expression of genes upstream of the perturbed gene are unlikely to be affected. |
| Motif enrichment | Welch t-test of per-gene TF motif counts (promoter / enhancer), top 300 program genes vs background (Schnitzler et al. 2024); alternative: correlation of motif counts with program × gene scores. See [Motif enrichment](#motif-enrichment) | MotifCompendium-Database-Human (FIMO, default) or HOCOMOCO v11, and/or ChromBPNet Fi-NeMo hits; enhancer–gene links (ABC / ENCODE-rE2G / IGVF scE2G) | If genes with high contributions to a program are also enriched with same enhancer/promoter motifs they could be co-regulated; candidate TFs are the expressed TFs of an enriched motif cluster that also load on / regulate the program. | Correlative. Many programs share a generic GC-rich SP/KLF/EGR/ZNF promoter signal. A motif cluster (e.g. KLF-SP_0) lists many TFs that cannot be told apart. |
| Trait enrichment | [Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) | OpenTargets database | If a program is significantly associated with a trait then it could explain the biological process the program represents. | |
| GO geneset enrichment | [GSEA](https://gseapy.readthedocs.io/en/latest/introduction.html) using program × feature scores | GO (Gene Ontology) | If a program is significantly associated with a GO term then it could explain the biological process the program represents. | |
| Geneset enrichment | [GSEA](https://gseapy.readthedocs.io/en/latest/introduction.html) using program × feature scores | MsigDB, Enrichr | If a program is significantly associated with a gene-set then it could explain the biological process the program represents. | |
| Explained variance | [Explained variance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html) per program | None | A program explaining more variance in the data might represent dominant biological variation. | Technical variation might be the highest source of variance (e.g. batch effects). |
| Reconstruction error | Frobenius norm of residual matrix | None | Lower reconstruction error indicates better overall fit of the NMF model to the data. | Lower error does not guarantee biologically meaningful programs. |
| Stability | Silhouette score of NMF replicate solutions | None | Higher stability indicates that programs are reproducible across NMF replicates. | High stability alone does not guarantee biological relevance. |

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| out_dir | str | Path to cNMF object directory |
| run_name | str | Name of cNMF object |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| K | list of int | [30, 50, 70, 80, 100, 200, 300] | K values to evaluate |
| sel_threshs | list of float | [0.2, 2.0] | Selection thresholds |
| X_normalized_path | str | None | Path to normalized cell x gene matrix (.h5ad), required for explained variance |
| guide_annotation_path | str | None | Path to tab-separated guide annotation file with "targeting" column |
| gwas_data_path | str | None | Path to GWAS data file for trait enrichment (required when --Perform_trait is set) |
| organism | str | "human" | Organism/species for enrichment analysis |
| FDR_method | str | "StoreyQ" | FDR correction method for perturbation association |
| n_top | int | 300 | Number of top loaded genes to use for enrichment tests |
| guide_annotation_key | list of str | ["non-targeting"] | Name(s) of non-targeting guide target labels |

### Data Access Keys

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_key | str | "rna" | Gene expression access key |
| prog_key | str | "cNMF" | cNMF program access key |
| categorical_key | str | "sample" | Cell condition key |
| guide_names_key | str | "guide_names" | Guide names key |
| guide_targets_key | str | "guide_targets" | Guide targets key |
| guide_assignment_key | str | "guide_assignment" | Guide assignment key |
| gene_names_key | str | "symbol" | Column in data_guide["rna"].var containing gene names |

### Analysis Flags

| Flag | Description |
|------|-------------|
| --Perform_categorical | Enable categorical association analysis |
| --Perform_perturbation | Enable perturbation association analysis |
| --Perform_geneset | Enable gene set enrichment analysis |
| --Perform_trait | Enable trait enrichment analysis |
| --Perform_explained_variance | Enable explained variance analysis |
| --Perform_motif | Enable TF motif enrichment + candidate-TF nomination (options below; same as `run_motif_enrichment.py`) |
| --use_cache | Load gene set libraries from cached JSON in `Resources/` instead of downloading; falls back to download + cache on miss |
| --skip_existing | Skip metric computations whose output files already exist on disk; useful for resuming preempted batches |
| --reassign_name | Reassign `mdata[data_key].var_names` from `var[gene_names_key]` before running metrics |


## Output Organization

```
Evaluation/
├── config_{job_id}.yml
└── {k}_{sel_thresh}/
    ├── {k}_categorical_association_results.txt
    ├── {k}_categorical_association_posthoc.txt
    ├── {k}_perturbation_association_results_{sample}.txt   (one per sample)
    ├── {k}_geneset_enrichment.txt
    ├── {k}_GO_term_enrichment.txt
    ├── {k}_trait_enrichment.txt
    ├── {k}_motif_enrichment.txt, {k}_candidate_tfs.txt, {k}_motif_logos.json, {k}_motif_enrichment_config.yml
    │   ({k}_finemo_pattern_names.tsv with Fi-NeMo)
    └── {k}_explained_variance*.txt
Evaluation/motif_hits/          cached promoter / enhancer motif hit tables (shared by all K)
```

## Motif enrichment

Code: `src/motif_enrichment.py` (statistics), `src/motif_hit_calling.py` (regions, FIMO, Fi-NeMo, MotifCompendium
metadata), `src/nominate_candidate_tfs.py` (candidate TFs), `src/motif_logos.py` (logo matrices),
`src/find_regulatory_resources.py` (portal search), driver `Slurm_Version/run_motif_enrichment.py` (+ `.sh`), also run by
`cNMF_evaluation_pipeline.py --Perform_motif`. `src/enrichment_motif.py` is the superseded tangermeme/correlation implementation.

Usage (defaults: MotifCompendium FIMO scan of strand-aware promoters + linked enhancers, t-test on the top 300 genes):

```bash
sbatch --export=ALL,PIPELINE_SRC=$PWD/Slurm_Version Slurm_Version/run_motif_enrichment.sh \
    --out_dir <out_dir> --run_name <run> --K <K> --sel_threshs 0.2 \
    --enhancer_links <E2G links (bedpe / tsv)> --genome_fasta <hg38.fa> --gene_annotation <genes.gtf.gz> \
    --motif_file <MotifCompendium-Database-Human.meme.txt> --fimo_binary "$(which fimo)"
# --genome_fasta / --gene_annotation / --motif_file default to $PERTURBNMF_GENOME_FASTA, $PERTURBNMF_GENE_ANNOTATION,
#   $PERTURBNMF_MOTIFCOMPENDIUM_MEME ($PERTURBNMF_HOCOMOCO_V11_MEME with --motif_db hocomoco_v11) when set
# + Fi-NeMo: --motif_source both --finemo_instances <instances tar/tsv> --finemo_report <report tar/html>
#   (or --finemo_annotation motif_annotation.tsv for local tables from export_motif_hits_for_perturbnmf.py)
# Schnitzler et al. 2024 setup (one test per HOCOMOCO v11 TF): --motif_db hocomoco_v11
```

### Method (defaults)

1. **Regions.** Promoters: one per gene symbol, TSS−250..TSS+50 in the direction of transcription (canonical
   transcript TSS from the GTF; from a BED6 of gene bounds, `start` (+) / `end − 1` (−)).
   `--promoter_window_mode schnitzler2024` = the strand-agnostic [TSS−250, TSS+51) of Schnitzler et al. 2024, with the BED
   `end` as the − gene TSS. Enhancers: element–gene links (`--enhancer_links`: ABC `Predictions`, ENCODE-rE2G /
   scE2G tsv, IGVF bedpe; promoter-class elements dropped -- for bedpe, elements within ±500 bp of their target TSS; `--link_score_threshold`), or the rank-1
   `e2g_links` row of `--regulatory_resources_manifest`.
2. **Motif database and hits.** `--motif_source fimo` (default): MEME `fimo` (e.g. 5.3.3; `--text`, so no
   `--max-stored-scores` truncation) or `memelite`; hits kept at p < 1e-4 (promoters) and p < 1e-6 (enhancers).
   `--motif_db motifcompendium` (default): MotifCompendium-Database-Human (926 non-redundant clusters merging JASPAR 2024,
   HOCOMOCO v12/v13, CIS-BP, Codebook and CAP-SELEX motifs); `hocomoco_v11`: the Schnitzler et al. 2024 database; or a MEME file path.
   `finemo`: ENCODE ChromBPNet/BPNet Fi-NeMo hit calls intersected with the same regions (no p-value filter); defaults counts
   head, lambda 0.7, `pos_patterns` only; each TF-MoDISco pattern is named by its top TOMTOM match in the report, a
   MotifCompendium cluster (`GATA_0`), whatever its q-value (`--finemo_qvalue_threshold` optionally keeps weaker patterns
   as `pos-counts-pattern-N`; the q-value is kept as `motif_match_qvalue`). Local Fi-NeMo tables (`--finemo_annotation`
   `motif_annotation.tsv` + `--finemo_instances motif_hits_<dataset>.tsv.gz`, from
   `Slurm_Version/export_motif_hits_for_perturbnmf.py`) name each motif by its `database_motif` and use its
   `candidate_tfs`. `both` runs FIMO and Fi-NeMo and adds `motif_source`.
   Hit tables are cached in `Evaluation/motif_hits/` (one folder per parameter set, `params.json`).
3. **Test unit and counts.** MotifCompendium (FIMO) and Fi-NeMo: one test per database motif cluster (`KLF-SP_0` and
   `KLF-SP_1` are separate rows; Fi-NeMo patterns with the same top match are pooled). HOCOMOCO / other MEME files:
   motif ids collapse to a TF (`KLF4_HUMAN.H11MO.0.A` → `KLF4`; ids without `_` use `motif_alt_id`, e.g. JASPAR
   `MA0139.1` → `CTCF`). Gene × motif counts summed over all enhancers of a gene. Universe = expressed genes (cNMF genes)
   with ≥ 1 hit.
4. **Test.** `--motif_method ttest` (default): top `--n_top` 300 genes per program vs the rest of the universe,
   Welch two-sided t-test, enrichment = mean ratio. `correlation`: Pearson / Spearman (`--motif_correlation`)
   of the gene's motif count with the program's full loading vector over all expressed genes (zeros for genes
   without hits); `enrichment` = r. BH across all program × motif per element type (and source).
   Significant = FDR < 0.05 and enrichment > 1 (t-test) or r > 0 (correlation).
5. **Motif families.** `motif_family` column: MotifCompendium cluster name without `_<n>` (`KLF-SP_0` → `KLF-SP`) for
   MotifCompendium FIMO and Fi-NeMo, so both sources share one vocabulary; for HOCOMOCO v11 the TFClass family
   (`Three-zinc finger Krüppel-related factors`) from the bundled annotation. Stage 3 (Excel, HTML, annotation viewer,
   annotator prompt) groups motifs by family. The test itself is not pooled by family.
6. **Candidate TFs.** Enriched motif → TF genes: the database's TF list of the cluster (bundled MotifCompendium metadata;
   HOCOMOCO mnemonics and hyphen-less names resolved) ∩ expressed genes, e.g. `GATA_0` → GATA1-6, GATAD2A, TAL1, TRPS1,
   ZFPM1; HOCOMOCO TFs via the HOCOMOCO annotation. Tiered: `motif+regulator` (knockdown changes the program, adj p < 0.05,
   from `{K}_perturbation_association_results_*.txt`) > `motif+expressed_in_program` (top-300 gene) >
   `motif+expressed` > `motif_only`.
7. **Logos.** `{K}_motif_logos.json`: for every motif significant in any program, an information-content matrix from the
   PFM (FIMO database; Fi-NeMo: the matched MotifCompendium PFM from `--finemo_pfm_file`) or the TF-MoDISco CWM
   (`--finemo_motifs`, ENCODE "sequence motifs" tar with the modisco h5). The annotation viewer draws them.

MotifCompendium releases: motif indices change between releases (`KLF_3` = KLF15 in 2025-09, KLF10/KLF11 from 2026-01),
so TF lists must come from the release the motifs were named with. Bundled (`src/motif_databases/`, trimmed columns):
`...metadata.2026-05-02_2ad26dc.tsv` = the current PFM file (default FIMO database) and `...metadata.2025-09-24_5b20d47.tsv` =
the release the ENCODE ChromBPNet TF-MoDISco reports were matched against. The release containing most of the motif names
is picked automatically (`--motifcompendium_metadata` overrides).

### Motif options (`run_motif_enrichment.py` and `cNMF_evaluation_pipeline.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| motif_method | ttest | `ttest` or `correlation` |
| motif_correlation | pearson | `pearson` or `spearman` (correlation method) |
| motif_element_types | promoter enhancer | element types to test |
| motif_source | fimo | `fimo`, `finemo` or `both` |
| motif_fdr_threshold | 0.05 | significance and candidate-TF FDR |
| motif_hit_cache_dir | `{out_dir}/{run_name}/Evaluation/motif_hits` | hit-table cache; safe to share between concurrent K jobs (tables built in a temporary dir, then renamed into place) |
| gene_annotation | `$PERTURBNMF_GENE_ANNOTATION` | GTF(.gz) or BED6 gene bounds for promoters (required for promoters) |
| promoter_window_mode | strand_aware | or `schnitzler2024` |
| promoter_upstream / promoter_downstream | 250 / 50 | strand-aware window |
| enhancer_links | None | element–gene links file (enhancers skipped if absent and no manifest) |
| regulatory_resources_manifest | None | `regulatory_resources_manifest.tsv`; best-ranked rows fill enhancer links / Fi-NeMo files (instances + report from the same `dataset_accession`) |
| link_format / link_score_threshold / link_score_column | auto / None / None | link parsing; a threshold without a score column is an error |
| merge_overlapping_links | off | merge overlapping enhancer elements of the same target gene into one region (links pooled over conditions / samples: a hit counts once per gene); also in `call_motif_hits.py` |
| genome_fasta | `$PERTURBNMF_GENOME_FASTA` | must match the build of GTF / links; `chr1` vs `1` naming is resolved; > 5% of intervals on chromosomes missing from the FASTA, or 0 FIMO hits, is an error |
| genome_build | hg38 | build of the FASTA and all coordinates; checked against the FASTA chr1 length, manifest `assembly` and build names in input file names (mismatch = error); part of the cache key. For hg19 inputs (e.g. hg19 ABC links) pass `--genome_build hg19 --genome_fasta hg19.fa`, or build hg19 tables with `call_motif_hits.py` and pass them as `--enhancer_hits` |
| motif_db | motifcompendium | FIMO database: `motifcompendium` (one test per cluster, TF lists from the database), `hocomoco_v11` (Schnitzler et al. 2024 setup, one test per TF), or a MEME file path (ids collapsed like HOCOMOCO) |
| motif_file | `$PERTURBNMF_MOTIFCOMPENDIUM_MEME` / `$PERTURBNMF_HOCOMOCO_V11_MEME` (per `--motif_db`) | MEME file to scan (names must follow `--motif_db`); precomputed hit tables with HOCOMOCO ids under the MotifCompendium default are an error |
| motifcompendium_metadata | bundled release with the most motif names | MotifCompendium metadata tsv (TF lists) |
| promoter_pvalue_threshold / enhancer_pvalue_threshold | 1e-4 / 1e-6 | FIMO hit p-value filters |
| fimo_backend / fimo_binary | auto / fimo | `meme` binary or `memelite`; the cache key stores the resolved backend, binary path and `fimo --version` |
| meme_default_mode | off | FIMO default mode (q-values, per-chunk `--max-stored-scores` cap) |
| n_chunks / n_jobs | 64 / 1 (`.sh`: CPUs) | FIMO parallelism |
| promoter_hits / enhancer_hits | None | precomputed FIMO tsv (skips scanning) |
| finemo_instances / finemo_report | None | Fi-NeMo tsv / report html, or the ENCODE tars / extracted dirs |
| finemo_annotation | None | local `motif_annotation.tsv` (motif_id, database_motif, candidate_tfs, posneg) instead of the report |
| finemo_motifs / finemo_pfm_file | None / MotifCompendium file if same release | logos: ENCODE "sequence motifs" tar (TF-MoDISco CWMs), else the matched database PFM |
| finemo_head / finemo_lambda | counts / 0.7 | which instance set |
| finemo_pattern_prefixes | pos_patterns | pattern types kept |
| finemo_qvalue_threshold | off | name a pattern by its top TOMTOM match only if q < this (default: always) |
| perturbation_results_path | run's `{K}_perturbation_association_results_*.txt` | knockdown evidence |
| knockdown_fdr_threshold | 0.05 | knockdown regulates the program (min adj_pval over the per-sample tables: significant in any sample) |
| motif_min_universe_genes | 1000 | error if fewer program genes have motif hits (Ensembl-id columns are first mapped to symbols via the GTF, or the h5mu `var[gene_names_key]`) |

`run_motif_enrichment.py` also takes `--out_dir --run_name --K --sel_threshs --n_top --skip_existing` and
`--gene_spectra_score_path` (override the programs table for a single K).

### Multi-condition screens

When a motif source is condition-specific (e.g. Fi-NeMo hits from a ChromBPNet model per condition), run
`run_motif_enrichment.py` once per condition (same programs; pooled links with `--merge_overlapping_links`),
then merge with `Slurm_Version/merge_motif_tables_at_peak_condition.py --K <K> --condition_runs A=<dir> B=<dir>
--peak_conditions <program,condition table> | --program_activity <program,condition,mean_score table>
--out_evaluation_dir <dir>`: each program keeps the condition-specific rows (`--per_condition_sources`, default
`finemo`) of its peak condition (table entry, else the condition with the highest mean score); other sources come from
the first run. Local Fi-NeMo / MotifCompendium outputs are converted to hit tables with
`Slurm_Version/export_motif_hits_for_perturbnmf.py`.

### Resources

| Resource | Download |
|----------|----------|
| MotifCompendium-Database-Human (default FIMO database; names Fi-NeMo patterns) | https://github.com/kundajelab/MotifCompendium (`pipeline/data/MotifCompendium-Database-Human.{meme.txt,metadata.tsv}`, MIT license, `src/motif_databases/MotifCompendium_LICENSE.txt`); method: https://zenodo.org/records/17179111. Metadata of commits 2ad26dc (2026-05-02) and 5b20d47 (2025-09-24, ENCODE reports) bundled in `src/motif_databases/` |
| HOCOMOCO v11 full human mono (MEME) + annotation | https://hocomoco11.autosome.org/downloads (`HOCOMOCOv11_full_HUMAN_mono_meme_format.meme`, `HOCOMOCOv11_full_annotation_HUMAN_mono.tsv`); annotation bundled in `src/motif_databases/` |
| JASPAR (alternative DB; names are gene symbols, dimers `A::B`) | https://jaspar.elixir.no/downloads/ (CORE vertebrates non-redundant, MEME format) |
| ENCODE ChromBPNet Fi-NeMo calls | encodeproject.org annotation sets "ChromBPNet models trained on DNase-seq ..." (files "sequence motifs instances" + "sequence motifs report") |
| Enhancer–gene links | IGVF scE2G; ENCODE-rE2G; ABC. Find with `src/find_regulatory_resources.py --cell-type <cell type> --out-dir <dir>` (optional `--synonyms <file.json>`, format in `src/resources/cell_type_synonyms.example.json`) |
| Genome / GTF | UCSC hg38 / hg19; a GENCODE or IGVF GTF |

See `src/motif_databases/README.md` for the sources and licenses of the bundled tables.

### Tests

`pytest --noconftest tests/Script/Stage2_Evaluation/test_motif_*.py tests/Script/Stage2_Evaluation/test_nominate_candidate_tfs.py tests/Script/Stage2_Evaluation/test_run_motif_enrichment.py tests/Script/Stage2_Evaluation/test_find_regulatory_resources.py tests/Script/Stage2_Evaluation/test_merge_motif_tables_at_peak_condition.py`
