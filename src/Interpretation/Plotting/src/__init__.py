# helper functions
from .utilities import convert_with_mygene, convert_adata_with_mygene, rename_adata_gene_dictionary, \
                        rename_list_gene_dictionary, read_npz, merge_pdfs_in_folder, merge_svgs_to_pdf


# K selection plots
from .k_selection_plots import load_stablity_error_data, plot_stablity_error,\
                               load_enrichment_data, plot_enrichment,\
                               load_perturbation_data, plot_perturbation,\
                               load_explained_variance_data,plot_explained_variance, plot_k_selection_panel, \
                                plot_k_selection_panel_no_traits


from .k_quality_plots import program_corr,program_euclidean, top_genes_overlap,sort_corr_matrix,\
                                programs_dotplots, consensus_clustermap, cNMF_boxplot, \
                                stability_vs_sharedgenes, cNMF_barplot, \
                                trait_clustermap, geneset_clustermap,perturbation_clustermap,\
                                GO_clustermap,build_overlap_matrix,compute_gene_list_GO,\
                                compute_gene_list_perturbation,load_combined_matrix,\
                                kmean_cluster, NMF_clustermap,plot_coefficient_variance,return_cNMF_matrix
                            

# gene QC plots
from .Perturbed_gene_QC_plots import  plot_umap_per_gene, plot_top_program_per_gene, perturbed_gene_dotplot,\
                                      plot_log2FC, plot_volcano, programs_dotplot, analyze_correlations, \
                                      create_comprehensive_plot,create_gene_correlation_waterfall,\
                                      plot_umap_per_gene_guide, process_single_gene, _process_gene_worker, parallel_gene_processing,\
                                      compute_gene_correlation_matrix, compute_gene_waterfall_cor, \
                                      plot_perturbation_vs_control, \
                                      plot_perturbation_vs_control_by_condition



# program QC plots
from .Program_QC_plots import plot_umap_per_program, plot_top_gene_per_program, top_GO_per_program, compute_program_correlation_matrix,\
                              analyze_program_correlations, plot_violin, plot_program_log2FC, plot_program_heatmap, plot_program_volcano, \
                              perturbed_program_dotplot, compute_program_waterfall_cor, create_program_correlation_waterfall, create_comprehensive_program_plot

# expression-weighted program plots
from .Program_expression_weighted_plots import compute_program_expression_by_condition, \
                                               plot_program_heatmap_weighted, \
                                               plot_program_heatmap_expression_scaled
