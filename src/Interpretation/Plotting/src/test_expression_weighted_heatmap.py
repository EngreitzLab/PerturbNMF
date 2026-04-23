#%%
import sys
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

import muon as mu
from Interpretation.Plotting.src.Program_expression_weighted_plots import (
    compute_program_expression_by_condition,
    plot_program_heatmap_weighted,
    plot_program_heatmap_expression_scaled,
)

#%% --- paths ---
mdata_path = "/oak/stanford/groups/engreitz/Users/ymo/Project/Morphic_MSK_KO/Result/MSK_KO_village_cNMF/Inference/outputs/cNMF_100_2.0.h5mu"
eval_dir   = "/oak/stanford/groups/engreitz/Users/ymo/Project/Morphic_MSK_KO/Result/MSK_KO_village_cNMF/Evaluation/100_2_0"

perturb_path_base = f"{eval_dir}/100_cNMF_gene_time_point_perturbation_association"

# pick a subset of time points to keep the plot readable
sample = ['D-1', 'D3', 'D7', 'D11', 'D18']

# program to plot (use string)
Target_Program = '10'

#%% --- load mdata ---
mdata = mu.read_h5mu(mdata_path)

#%% --- 1. check program expression by condition ---
expr = compute_program_expression_by_condition(mdata, Target_Program, groupby='time_point')
print("Mean program expression per time point:")
print(expr.loc[sample])

#%% --- 2. heatmap with expression side bar ---
ax = plot_program_heatmap_weighted(
    perturb_path_base=perturb_path_base,
    mdata=mdata,
    Target_Program=Target_Program,
    tagert_col_name='program_name',
    plot_col_name='target_name',
    sample=sample,
    groupby='time_point',
    log2fc_col='log2FC',
    p_value=0.05,
    figsize=(14, 5),
    show=True,
    vmin=-1, vmax=1,
)
ax.figure.savefig(f"{eval_dir}/program_{Target_Program}_heatmap_weighted.png", dpi=150, bbox_inches='tight')
print(f"Saved: {eval_dir}/program_{Target_Program}_heatmap_weighted.png")


#%% --- 3. expression-scaled heatmap ---
ax2 = plot_program_heatmap_expression_scaled(
    perturb_path_base=perturb_path_base,
    mdata=mdata,
    Target_Program=Target_Program,
    tagert_col_name='program_name',
    plot_col_name='target_name',
    sample=sample,
    groupby='time_point',
    log2fc_col='log2FC',
    p_value=0.05,
    figsize=(14, 5),
    show=False,
)
ax2.figure.savefig(f"{eval_dir}/program_{Target_Program}_heatmap_expression_scaled.png", dpi=150, bbox_inches='tight')
print(f"Saved: {eval_dir}/program_{Target_Program}_heatmap_expression_scaled.png")
# %%
