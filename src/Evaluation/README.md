

# cNMF Evaluation Pipeline

Comprehensive evaluation metrics for cNMF program quality assessment across multiple biological and technical dimensions.

## Overview

The evaluation pipeline tests cNMF programs against various criteria to assess their biological relevance and technical quality. Each program is systematically evaluated using statistical tests and enrichment analyses.

## Evaluation Criteria

| Criterion    | Implementation | External resource | Interpretation | Caveats |
| -------- | ------- | -------- | ------- | ------- |
| Categorical association | [Kruskal-Wallis non-parametric ANOVA](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance) + Dunn's posthoc test | None | If program scores are variable between batch levels then the component likely is modelling technical noise. Alternatively, if program scores are variable between a biological category like cell-type or condition then the program is likely modelling a biological process specific to the category. | If batches are confounded with biological conditions, then the relative contribution of technical and biological variation cannot be decomposed. |
| Perturbation sensitivity | Mann-Whitney U test of program scores between perturbed cells and non-targeted/reference cells | Perturbation data | Cell × program score distribution shifts greater than expected due to the direct effect of perturbation on genes in the program could indicate hierarchical relationships between genes in the program. | Expression of genes upstream of the perturbed gene are unlikely to be affected. |
| Motif enrichment | Pearson correlation of motif counts per gene (promoter or enhancer) and program × gene scores | HOCOMOCO v12 | If genes with high contributions to a program are also enriched with same enhancer/promoter motifs they could be co-regulated. | A biological pathway could involve genes with different regulation but still contribute to a common function. |
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
| gwas_data_path | str | Path to GWAS information file |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| K | list of int | [30, 50, 60, 80, 100, 200, 250, 300] | K values to evaluate |
| sel_thresh | list of float | [0.4, 0.8, 2.0] | Selection thresholds |
| X_normalized_path | str | None | Path to normalized cell x gene matrix (.h5ad), required for explained variance |
| data_guide_path | str | None | Path to MuData (.h5mu) containing additional guide/gene information |
| guide_annotation_path | str | None | Path to tab-separated guide annotation file with "targeting" column |
| reference_gtf_path | str | None | Path to reference GTF file |
| organism | str | "human" | Organism/species for enrichment analysis |
| FDR_method | str | "StoreyQ" | FDR correction method for perturbation association |
| n_top | int | 300 | Number of top loaded genes to use for enrichment tests |
| check_format | flag | False | Validate MuData format before running evaluation |
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

### Analysis Flags

| Flag | Description |
|------|-------------|
| --Perform_categorical | Enable categorical association analysis |
| --Perform_perturbation | Enable perturbation association analysis |
| --Perform_geneset | Enable gene set enrichment analysis |
| --Perform_trait | Enable trait enrichment analysis |
| --Perform_explained_variance | Enable explained variance analysis |
| --Perform_motif | Enable motif enrichment analysis |


## Output Organization

```
├── Evaluation/
├── k_density_threshold/ 
├── k_categorical_association_posthoc.csv 
├── k_categorical_association_results.csv 
├── k_explained_variance_summary.csv 
├── k_explained_variance.csv 
├── k_geneset_enrichment.csv 
├── k_go_term_enrichment.csv 
├── k_perturbation_association_d0.csv 
├── k_perturbation_association_d1.csv 
├── k_perturbation_association_d2.csv 
├── k_perturbation_association_d3.csv 
└── k_trait_enrichment.csv
```