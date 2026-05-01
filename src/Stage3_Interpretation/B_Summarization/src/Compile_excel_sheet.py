import pandas as pd
import muon as mu
from scipy import sparse
import sys
import os
import numpy as np
from scipy import stats
from mygene import MyGeneInfo
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from functools import reduce



# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')


from Interpretation.Plotting.src import rename_adata_gene_dictionary ,compute_gene_waterfall_cor

#-------------- helper methods compile into excel sheets, not used here --------------

# check matching names in program
def check_program_name_match(mdata, dataframes, prog_key='cNMF'):
    """Check that program names in all loaded DataFrames match mdata var_names.

    Parameters
    ----------
    mdata : MuData
        The MuData object containing cNMF results.
    dataframes : list of DataFrame or None
        List of DataFrames to check. None values are skipped.
        Each DataFrame is expected to have a 'program_name' column.
    prog_key : str
        Key for the program modality in mdata.
    """
    mdata_programs = set(str(v) for v in mdata[prog_key].var_names)
    try:
        mdata_sorted = sorted(mdata_programs, key=int)
    except ValueError:
        mdata_sorted = sorted(mdata_programs)

    mismatched = []
    for df in dataframes:
        if df is None:
            continue
        if 'program_name' not in df.columns:
            continue
        file_programs = set(str(v) for v in df['program_name'].unique())
        if file_programs != mdata_programs:
            try:
                file_sorted = sorted(file_programs, key=int)
            except ValueError:
                file_sorted = sorted(file_programs)
            mismatched.append(file_sorted)

    if mismatched:
        print(f"WARNING: Program name mismatch detected!")
        print(f"  mdata['{prog_key}'].var_names: {mdata_sorted}")
        for file_sorted in mismatched:
            print(f"  file programs: {file_sorted}")

# merge to have specificity score added for perturbation results
def add_specificity_scores_file(save_path, Perturbation_path_base, samp):                                                                                                                                                    
                                                                                                                                                                                                                     
      PMI = pd.read_csv(f'{save_path}/specificity_score_{samp}.txt', sep='\t', index_col=0)                                                                                                                          
      df_perturbation = pd.read_csv(f'{Perturbation_path_base}_{samp}.txt', sep='\t')                                                                                                                                     
                                                                                                                                                                                                                     
      # convert PMI to long form
      df_PMI = PMI.reset_index().melt(id_vars="target_name", var_name="program_name", value_name="specificity_scores")

      # make sure program name is in str form
      df_PMI["program_name"] = df_PMI["program_name"].astype(str)
      df_perturbation["program_name"] = df_perturbation["program_name"].astype(str)

      # merge to have specificity score added for perturbation results
      df_perturbation_merged = df_PMI.merge(df_perturbation, on=["target_name", "program_name"])

      df_perturbation_merged.to_csv(f'{save_path}/perturbation_merged_{samp}.txt', sep='\t')

      return df_perturbation_merged




# Compile simple sheets

#-------------- helper methods for loading sheets--------------
def compile_Program_loading_score_sheet_long(mdata, num_gene = 300):

    print('Load program loadings data in long form')

    program_loading_df = pd.DataFrame(data=mdata['cNMF'].varm["loadings"], columns=mdata['cNMF'].uns['var_names'], index=mdata['cNMF'].var_names)

    top_df = program_loading_df.apply(
    lambda row: row.nlargest(num_gene).index.tolist(),
    axis=1
    )

    result_df = pd.DataFrame(top_df.tolist(), columns=range(1, num_gene+1))
    result_df.index = program_loading_df.index

    result_df.index.name = "Program"

    mg = MyGeneInfo()
    long_data = []

    for program_idx, row in result_df.iterrows():

        # Query all genes for this program at once
        genes_list = row.dropna().astype(str).str.strip().tolist()
        genes_list = [g for g in genes_list if g]  # Remove empty strings
        
        if not genes_list:
            continue
        
        # Get results for all genes in this program
        results = mg.querymany(genes_list, scopes='symbol', fields='summary', species='human', verbose=False)
        
        # Create a lookup dictionary for faster access
        summary_dict = {}
        for r in results:
            q = r.get('query')
            if q not in summary_dict:  # keep first (best) match
                summary_dict[q] = r.get('summary', 'N/A')
        
        # Now iterate through the ranked genes
        for rank, gene in enumerate(row, 1):
            
            # Skip NaN values
            if pd.notna(gene) and gene != '':
                gene_clean = str(gene).strip()
                annotation = summary_dict.get(gene_clean, 'N/A')
                
                long_data.append({
                    'Program': program_idx,
                    'Rank': rank,
                    'Gene': gene_clean,
                    'Annotation': annotation
                })

    # Convert to DataFrame
    annotation_df = pd.DataFrame(long_data)

    return annotation_df

def compile_Program_loading_score_sheet_flat(mdata, num_gene = 300):

    print('Load program loadings data in flat form')


    program_loading_df = pd.DataFrame(data=mdata['cNMF'].varm["loadings"], columns=mdata['cNMF'].uns['var_names'], index=mdata['cNMF'].var_names)

    top_df = program_loading_df.apply(
    lambda row: row.nlargest(num_gene).index.tolist(),
    axis=1
    )

    result_df = pd.DataFrame(top_df.tolist(), columns=range(1, num_gene+1))
    result_df.index = program_loading_df.index

    result_df.index.name = "Program"

    return result_df

def Compile_GO_sheet(GO_path, gene_num = 5, term_key = "Term", genes_key = "Genes"):

    print('Load GO data')

    df = pd.read_csv(GO_path, sep = "\t", index_col = 0)
    df = df.reset_index().set_index(term_key)

    df[genes_key] = df[genes_key].str.split(';').str[:gene_num].str.join(";")

    return df

def Compile_Geneset_sheet(Geneset_path, gene_num = 5, term_key = "Term", genes_key = "Genes"):

    print('Load geneset data')


    df = pd.read_csv(Geneset_path, sep = "\t", index_col = 0)
    df = df.reset_index().set_index(term_key)

    df[genes_key] = df[genes_key].str.split(';').str[:gene_num].str.join(";")

    return df

def Compile_Trait_sheet(Trait_path, gene_num = 5, term_key = "Term", genes_key = "Genes"):

    print('Load trait data')

    df = pd.read_csv(Trait_path, sep = "\t", index_col = 0)
    df = df.reset_index().set_index(term_key)

    df[genes_key] = df[genes_key].str.split(';').str[:gene_num].str.join(";")

    return df

def Compile_Perturbation_sheet(Perturbation_path, Sample = ["D0", "sample_D1","sample_D2","sample_D3" ], sample_key = "Sample"):

    print('Load perturbation data')


    combined_conditions = []
    for samp in Sample:
        df = pd.read_csv(f"{Perturbation_path}_{samp}.txt", sep = "\t")
        df[sample_key] = samp
        combined_conditions.append(df)

    df = pd.concat(combined_conditions)
    
    return  df

def Compile_Association_sheet(Association_path, gene_num = 5):

    print('Load categorical associa data')

    df = pd.read_csv(Association_path, sep = "\t", index_col = 0)
    df = df.reset_index(drop=True)
    df.index.name = "program_name"
    df.sort_values(by = df.index.name, inplace = True, ascending=True)

    return df

def Compile_Explained_variance(Explained_Variance_path):

    print('Load explained variance data')

    df = pd.read_csv(Explained_Variance_path, sep = "\t")
    df = df.set_index('program_name') 

    return df

#-------------- helper methods for loading sheets--------------

# combine methods for simple sheets
def load_simple_sheets(mdata, out_dir, run_name, k, sel_thresh, num_gene = 300, perturbation_file_name = "perturbation_association_results", Sample = ['1', '2', '3'],
                       GO_Term_key = "Term", GO_Genes_key = "Genes",
                       Geneset_Term_key = "Term", Geneset_Genes_key = "Genes",
                       Trait_Term_key = "Term", Trait_Genes_key = "Genes",
                       Perturbation_Sample_key = "Sample"):

    print('Load simple sheet')

    GO_path = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_GO_term_enrichment.txt'
    Geneset_path = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_geneset_enrichment.txt'
    Trait_path = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_trait_enrichment.txt'
    Perturbation_path_base = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_{perturbation_file_name}'
    Association_path = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_categorical_association_results.txt'
    Explained_Variance_path = f'{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace(".","_")}/{k}_Explained_Variance.txt'

    # compile program loadings
    df_Program_loading_long = compile_Program_loading_score_sheet_long(mdata, num_gene = num_gene)
    df_Program_loading_flat = compile_Program_loading_score_sheet_flat(mdata, num_gene = num_gene)

    # compile GO
    if os.path.exists(GO_path):
        df_GO = Compile_GO_sheet(GO_path, gene_num = num_gene, term_key = GO_Term_key, genes_key = GO_Genes_key)
    else:
        print(f'GO file not found: {GO_path}')
        df_GO = None

    # compile Genesets
    if os.path.exists(Geneset_path):
        df_Geneset = Compile_Geneset_sheet(Geneset_path, gene_num = num_gene, term_key = Geneset_Term_key, genes_key = Geneset_Genes_key)
    else:
        print(f'Geneset file not found: {Geneset_path}')
        df_Geneset = None

    # compile Trait
    if os.path.exists(Trait_path):
        df_Trait = Compile_Trait_sheet(Trait_path, gene_num = num_gene, term_key = Trait_Term_key, genes_key = Trait_Genes_key)
    else:
        print(f'Trait file not found: {Trait_path}')
        df_Trait = None

    # compile perturbation
    perturbation_files = [f"{Perturbation_path_base}_{samp}.txt" for samp in Sample]
    if any(os.path.exists(f) for f in perturbation_files):
        df_Perturbation = Compile_Perturbation_sheet(Perturbation_path_base, Sample = Sample, sample_key = Perturbation_Sample_key)

        df_Perturbation_significant_gene_only = df_Perturbation[df_Perturbation['adj_pval'] < 0.05]
    else:
        print(f'No perturbation files found for base: {Perturbation_path_base}')
        df_Perturbation = None
        df_Perturbation_significant_gene_only = None

    # compile association
    if os.path.exists(Association_path):
        df_Association = Compile_Association_sheet(Association_path, gene_num = num_gene)
    else:
        print(f'Association file not found: {Association_path}')
        df_Association = None

    # compile explained variance
    if os.path.exists(Explained_Variance_path):
        df_Explained_Variance = Compile_Explained_variance(Explained_Variance_path)
    else:
        print(f'Explained variance file not found: {Explained_Variance_path}')
        df_Explained_Variance = None

    return df_Program_loading_long, df_Program_loading_flat, df_GO, df_Geneset, df_Trait, df_Perturbation, df_Association, df_Explained_Variance, df_Perturbation_significant_gene_only





# calcaulte specicicity scores

#-------------- helper methods for specicicity scores--------------

# helper methods for specicicity score calcualtion 
def Compute_regulator_zscore(perturb_df, effect_size='log2FC'):

        # Pivot to get program × gene log2FC table
        pivot_table = perturb_df.pivot_table(
            index='target_name',
            columns='program_name',
            values=effect_size,
            aggfunc='first'
        )

        # convert to abs values
        # pivot_table_abs = pivot_table.abs()

        # Compute z-scores row-wise
        median = pivot_table.median(axis=1)
        mad = pivot_table.sub(median, axis=0).abs().median(axis=1) 
        epsilon = 1e-10

        z_score_table = pivot_table.sub(median, axis=0).div(1.4826 * mad + epsilon, axis=0) 
        
        return z_score_table

def softmax_with_temperature(effects_df, T=1.0):

    # eq = e^{z/T}/ sum[ e^{z/T}]
    
    # Convert pandas to numpy and abs
    effects = effects_df.abs().values
    
    effects_scaled = effects / T
    effects_scaled = effects_scaled - effects_scaled.max(axis=1, keepdims=True) # for overflow prevention 
    
    exp_effects = np.exp(effects_scaled)
    weights = exp_effects / exp_effects.sum()
    
    # Convert back to DataFrame
    return pd.DataFrame(weights, 
                       index=effects_df.index, 
                       columns=effects_df.columns)

def compute_joint_distribution(weights):

    weights_array = weights.values
    
    # Joint: divide entire matrix by total sum
    P_tp = weights_array / weights_array.sum()
    
    # Marginals: sum by rows/columns
    P_t = P_tp.sum(axis=1)  # Row sums
    P_p = P_tp.sum(axis=0)  # Column sums

    # Convert back to DataFrame
    P_tp = pd.DataFrame(P_tp, 
                       index=weights.index, 
                       columns=weights.columns)

    
    return P_tp, P_t, P_p

def compute_pointwise_mutual_information(P_tp, P_t, P_p):

    # Reshape marginals for broadcasting
    P_t_col = P_t.reshape(-1, 1)  # Column vector
    P_p_row = P_p.reshape(1, -1)  # Row vector
    
    # P(t) * P(p) for all combinations
    P_independent = P_t_col @ P_p_row  # Outer product
    
    # Avoid log(0)
    epsilon = 1e-10
    
    # PMI = log(P_tp / P_independent)
    PMI = np.log((P_tp.values + epsilon) / (P_independent + epsilon))
    
    # Convert back to DataFrame for readability
    PMI_df = pd.DataFrame(PMI, 
                    index=P_tp.index, 
                    columns=P_tp.columns)
    
    return PMI_df

#-------------- helper methods specicicity scores--------------

# get specific programs and scores 
def get_specificity_program(Perturbation_path, Sample = ["D0", "sample_D1", "sample_D2", "sample_D3"], T=0.5, save_path = 
  None, effect_size='log2FC'):                                                                                                  
                  
      print("Calculate specificity scores for program")                                                                         
                  
      all_specificity = []

      for samp in Sample:
          df = pd.read_csv(f"{Perturbation_path}_{samp}.txt", sep="\t")
          df['program_name'] = df['program_name'].astype(str)

          z_score_table = Compute_regulator_zscore(df, effect_size=effect_size)
          weights = softmax_with_temperature(z_score_table, T=T)
          P_tp, P_t, P_p = compute_joint_distribution(weights)
          PMI = compute_pointwise_mutual_information(P_tp, P_t, P_p)

          if save_path is not None:
              PMI.to_csv(f'{save_path}/specificity_score_{samp}.txt', sep = '\t')

          # Fix #3: Ensure PMI columns are strings to match sig_mask columns
          PMI.columns = PMI.columns.astype(str)

          # Fix #4: Deduplicate PMI index to prevent .loc returning a DataFrame
          PMI = PMI[~PMI.index.duplicated(keep='first')]

          # Build a mask of significant gene-program pairs (adj_pval < 0.1)
          sig_mask = df[df['adj_pval'] < 0.1].pivot_table(
              index='target_name', columns='program_name', values='adj_pval', aggfunc='first'
          ).notna()

          # Mask PMI: keep only significant target-program pairs, NaN the rest
          sig_mask_aligned = sig_mask.reindex(index=PMI.index, columns=PMI.columns, fill_value=False)
          PMI_filtered = PMI.where(sig_mask_aligned)

          # For each target, get top 5 programs by PMI score
          program_specificity = []
          for gene_name in PMI_filtered.index:
              top_5 = PMI_filtered.loc[gene_name].dropna().nlargest(5)
              top_5_programs = ', '.join(top_5.index.astype(str)) if len(top_5) > 0 else ''
              top_5_scores = ', '.join([f"{score:.4f}" for score in top_5.values]) if len(top_5) > 0 else ''

              program_specificity.append({
                  'target_name': gene_name,
                  f'top 5 specific programs (FDR < 0.1) {samp}': top_5_programs,
                  f'top 5 specificity scores (FDR < 0.1) {samp}': top_5_scores,
              })

          df_specificity = pd.DataFrame(program_specificity).set_index('target_name')
          all_specificity.append(df_specificity)

      # Fix #5: Guard against empty Sample list
      if not all_specificity:
          return pd.DataFrame()

      # Merge all results by index (gene)
      merged_df = pd.concat(all_specificity, axis=1)

      return merged_df




# load target summary

#-------------- helper methods target summary--------------

# Get cells with the guide per day
def get_guide_cells_per_days(mdata, categorical_key="sample", data_key='rna', prog_key='cNMF',
                            guide_assignment_key="guide_assignment", guide_targets_key="guide_targets"):
                                                                                                                                                                                                                                                    
    print('Compute guides cells per days')
                                                                                                                                                                                                                                                    
    guide_assignment = mdata[prog_key].obsm[guide_assignment_key]  # (n_cells × n_targets), no copy                                                                                                                                              
    guide_targets = mdata[prog_key].uns[guide_targets_key]

    if not sparse.issparse(guide_assignment):
        guide_assignment = sparse.csc_matrix(guide_assignment)

    # Build sparse group indicator matrix (n_cells × n_groups)
    groups = mdata[data_key].obs[categorical_key]
    unique_groups = sorted(groups.unique())
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    col_idx = groups.map(group_to_idx).values.astype(np.int32)

    indicator = sparse.csc_matrix(
        (np.ones(len(groups), dtype=np.float32), (np.arange(len(groups)), col_idx)),
        shape=(len(groups), len(unique_groups))
    )

    # Sparse matmul: (n_groups × n_cells) @ (n_cells × n_targets) = (n_groups × n_targets)
    group_sums = indicator.T @ guide_assignment

    # Build result: (n_targets × n_groups)
    df_merge = pd.DataFrame(
        np.asarray(group_sums.todense()).T,
        index=guide_targets,
        columns=[f'# Cells {g}' for g in unique_groups]
    )

    # Collapse duplicate target names
    df_merge = df_merge.groupby(df_merge.index).sum()
    df_merge.index.name = "target_name"

    return df_merge

# Get the gene expression average per day 
def get_guide_mean_expr_per_day(mdata, categorical_key = "sample", prog_key = 'cNMF', data_key = 'rna', guide_targets_key = "guide_targets"):
                                                                                                                                                                            
    print('Get targeted gene mean expressuion per day')                                                                                                                                      
                                                                                                                                                                            
    adata = mdata[data_key]  # no .copy() — avoid duplicating the whole object                                                                                           

    perturbed_genes = mdata[prog_key].uns[guide_targets_key]

    gene_mask = adata.var_names.isin(perturbed_genes)
    gene_names = adata.var_names[gene_mask]
    X_sub = adata[:, gene_mask].X  # keep sparse, only subset columns

    if not sparse.issparse(X_sub):
        X_sub = sparse.csc_matrix(X_sub)

    # Compute mean per group directly on the sparse matrix
    groups = adata.obs[categorical_key]
    mean_per_group = {}
    for group_name in groups.unique():
        row_mask = (groups == group_name).values
        mean_per_group[f'mean_expression_{group_name}'] = np.asarray(X_sub[row_mask].mean(axis=0)).ravel()

    mean_expr_per_day = pd.DataFrame(mean_per_group, index=gene_names)
    mean_expr_per_day.sort_index(inplace=True)
    mean_expr_per_day.index.name = 'target_name'

    return mean_expr_per_day

# Function to get significant programs for each gene across all days
def get_significant_programs(Perturbation_path, Sample = ["D0", "sample_D1","sample_D2","sample_D3"], adj_pval_threshold=0.05, effect_size='log2FC'):

    print('Compute significant programs')
    
    significant_programs = {}
    
    for samp in Sample:
        # Read perturbation results for each day
        df = pd.read_csv(f"{Perturbation_path}_{samp}.txt", sep="\t")
        
        # Filter for significant associations
        significant = df[df['adj_pval'] < adj_pval_threshold]
        
        # Group by target_name and collect significant programs
        for target in significant['target_name'].unique():
            target_data = significant[significant['target_name'] == target]
            programs = target_data['program_name'].tolist()
            
            if target not in significant_programs:
                significant_programs[target] = {}
            
            significant_programs[target][samp] = programs
    
    return significant_programs

# get signifcant programs 
def get_significant_programs_df(Perturbation_path, Sample = ["D0", "sample_D1","sample_D2","sample_D3"], adj_pval_threshold=0.05, effect_size='log2FC'):

    print('Compile significant programs')

    # Get significant programs
    significant_programs = get_significant_programs(Perturbation_path, Sample, adj_pval_threshold, effect_size=effect_size)

    # Convert to DataFrame format for easier viewing
    sig_prog_data = []

    for gene, days_data in significant_programs.items():
        row = {'target_name': gene}
        for samp in Sample:
            programs = days_data.get(samp, [])
            row[f'significant programs {samp}'] = ', '.join(map(str, programs)) if programs else ''
        sig_prog_data.append(row)

    df_significant_programs = pd.DataFrame(sig_prog_data)
    df_significant_programs = df_significant_programs.set_index('target_name')

    for samp in Sample:
        df_significant_programs[f'# programs {samp}'] = df_significant_programs[f'significant programs {samp}'].apply(
            lambda x: str(len(x.split(','))) if x and x != '' else 0
        )


    return df_significant_programs

# Get top correlation terms 
def get_correlation_df(perturbation_path, Sample=["D0", "sample_D1", "sample_D2", "sample_D3"], top_n=5, save_path = None, effect_size='log2FC'):

    correlation_results_all_days = {}

    for day in Sample:
        print(f"Compute correlation for {day}")
        perturb_path = f"{perturbation_path}_{day}.txt"

        # Read and pivot directly
        df = pd.read_csv(perturb_path, sep='\t', index_col=0)
        pivot_df = df.pivot_table(index='target_name', columns='program_name', values=effect_size)

        # Compute correlation matrix (genes x genes) — vectorized
        corr_matrix = pivot_df.T.corr()

        # Zero out diagonal so self-correlation doesn't appear in top/bottom
        np.fill_diagonal(corr_matrix.values, np.nan)

        if save_path is not None:
            corr_matrix.to_csv(f"{save_path}/corr_gene_matrix_{day}.txt", sep="\t")
            corr_matrix.to_csv(f"{save_path}/corr_gene_matrix_{day}.txt.gz", sep="\t", compression="gzip")

        genes = corr_matrix.index.values
        vals = corr_matrix.values  # numpy array

        # Use argpartition for O(n) top/bottom selection instead of full sort
        rows = []
        for i in range(len(genes)):

            # for each row, remove NaN and get valid genes 
            row = vals[i]
            valid = ~np.isnan(row)
            valid_genes = genes[valid]

            # Top positive
            top_idx = np.argpartition(row[valid], -(top_n-1))[-top_n:] # get top 5 value's index (-1 for accessing k-th value)
            top_idx = top_idx[np.argsort(row[valid][top_idx])[::-1]]   # get top 5 values and sort them in right otder 

            # Bottom negative
            bot_idx = np.argpartition(row[valid], (top_n-1))[:top_n]  # get bottom 5 value's index (-1 for accessing k-th value)
            bot_idx = bot_idx[np.argsort(row[valid][bot_idx])]        # get bottom 5 values and sort them in right otder 

            rows.append({
                'target_name': genes[i],
                f'top 5 pos correls targets (program log2fc) {day}': '; '.join(str(g) for g in valid_genes[top_idx]),
                f'top 5 pos correls (program log2fc) {day}': '; '.join(f"{v:.3f}" for v in row[valid][top_idx]),
                f'top 5 neg correls targets (program log2fc) {day}': '; '.join(str(g) for g in valid_genes[bot_idx]),
                f'top 5 neg correls (program log2fc) {day}': '; '.join(f"{v:.3f}" for v in row[valid][bot_idx]),
            })

        correlation_results_all_days[day] = pd.DataFrame(rows).set_index('target_name')



    return pd.concat(correlation_results_all_days.values(), axis=1)

#-------------- helper methods starget summary--------------

# final function to compile target Summary sheet 
def Compile_Target_Summary_sheet(mdata, perturbation_path, Sample = ["D0", "sample_D1","sample_D2","sample_D3"], adj_pval_threshold= 0.05,
top_n=5, T= 0.5 , categorical_key = "sample", prog_key = 'cNMF', data_key = 'rna', guide_targets_key = "guide_targets", guide_assignment_key = "guide_assignment", save_path = None, effect_size='log2FC'):

    print('Generate target summary sheet')


    df_mean_expr_per_day = get_guide_mean_expr_per_day(mdata, categorical_key=categorical_key, prog_key=prog_key, data_key=data_key, 
    guide_targets_key=guide_targets_key)
    df_guide_days = get_guide_cells_per_days(mdata,  categorical_key=categorical_key, guide_assignment_key=guide_assignment_key, prog_key=prog_key, 
    data_key=data_key, guide_targets_key=guide_targets_key)
    df_significant_program = get_significant_programs_df(perturbation_path ,Sample=Sample, adj_pval_threshold=adj_pval_threshold, effect_size=effect_size)
    df_specificity_program =  get_specificity_program(perturbation_path, Sample=Sample, T = T, save_path = save_path, effect_size=effect_size) # save_path for saving specificity scores
    df_correlation = get_correlation_df(perturbation_path ,Sample=Sample, top_n=top_n, save_path = save_path, effect_size=effect_size)


    dfs = [df_mean_expr_per_day, df_guide_days, df_significant_program, df_specificity_program, df_correlation]

    final_merged_df = reduce(
        lambda left, right: left.merge(right, left_index=True, right_index=True, how='outer'),
        dfs
    ).fillna('')
    
    return final_merged_df





# load summary

#-------------- helper methods summary--------------

# get simply items ready in the summary sheet
def simple_Summary_cols(df, k, df_GO, df_Perturbation, df_Program_loading, df_Explained_Variance = None, specicicity_path = None, Sample = ["D0",
  "sample_D1","sample_D2","sample_D3" ],
  non_tagerting_key = None, effect_size='log2FC', adjusted_pval_key='Adjusted P-value'):                                                                                                   
   
      programs = list(df.index)  # use actual program indices from df, not range(k)                                                                  
                  
      # Fix #2: Cast programs to str for consistent comparisons with program_name columns
      programs_str = [str(p) for p in programs]

      # create GO summary col
      if df_GO is not None:
          df_GO_enriched = df_GO.loc[df_GO[adjusted_pval_key]<=0.05]
          df['Total Enriched GO Terms'] = [df_GO_enriched[df_GO_enriched['program_name'].astype(str)==i].shape[0] for i in programs_str]

      # remove non-targeting off the list of perturbed genes
      if df_Perturbation is not None:
          if non_tagerting_key is not None:
              df_Perturbation = df_Perturbation[~df_Perturbation['target_name'].isin(non_tagerting_key)]

          # Fix #2: Cast program_name to str for consistent comparisons
          df_Perturbation = df_Perturbation.copy()
          df_Perturbation['program_name'] = df_Perturbation['program_name'].astype(str)

          # compute # regulator per condition for + and -
          conditions = (df_Perturbation['Sample']).unique()
          for condition in conditions:
              df_Perturbation_ = df_Perturbation[df_Perturbation['Sample'] == condition]

              # create perturbation program summary col
              df_Perturbation_enriched = df_Perturbation_.loc[df_Perturbation_['adj_pval']<=0.05]
              df_Perturbation_positive = df_Perturbation_enriched.loc[df_Perturbation_enriched[effect_size] > 0 ]
              df_Perturbation_negative = df_Perturbation_enriched.loc[df_Perturbation_enriched[effect_size] < 0 ]

              df[f'Significant regulators with positive effect {condition}'] = [df_Perturbation_positive[df_Perturbation_positive['program_name']==i].shape[0] for i in programs_str]
              df[f'Significant regulators with negative effect {condition}'] = [df_Perturbation_negative[df_Perturbation_negative['program_name']==i].shape[0] for i in programs_str]

          # create perturbation gene summary col
          for condition in conditions:
              df_Perturbation_enriched = df_Perturbation.loc[df_Perturbation['adj_pval']<=0.05]
              df_Perturbation_D = df_Perturbation_enriched.loc[df_Perturbation_enriched['Sample'] == condition]

              targets_list = []
              for i in programs_str:
                  matching = df_Perturbation_D.loc[df_Perturbation_D['program_name'] == i]
                  unique_indices = matching["target_name"].unique()
                  joined = ';'.join([str(x) for x in unique_indices])
                  targets_list.append(joined)

              df[f'sigfdr0.05_targets_sorted_abslog2fcd_{condition}'] = targets_list

      # Fix #5: Moved specificity block outside df_Perturbation guard
      if specicicity_path is not None:
          if df_Perturbation is not None:
              conditions = df_Perturbation['Sample'].unique()
          else:
              conditions = Sample

          for condition in conditions:
              specificity_score_df = pd.read_csv(os.path.join(specicicity_path, f"specificity_score_{condition}.txt"), sep="\t", index_col=0)

              col = {}
              for prog in specificity_score_df.columns:
                  if df_Perturbation is not None:
                      df_Perturbation_cond = df_Perturbation[df_Perturbation['Sample'] == condition]
                      prog_perturb = df_Perturbation_cond[df_Perturbation_cond['program_name'] == (prog)]
                      sig_genes = prog_perturb[prog_perturb['adj_pval'] < 0.1]['target_name'].values
                      filtered_scores = specificity_score_df.loc[specificity_score_df.index.isin(sig_genes), prog]
                  else:
                      filtered_scores = specificity_score_df[prog].dropna()

                  top5 = filtered_scores.nlargest(5).index.tolist()
                  # Fix #3: Cast key to match df.index dtype
                  try:
                      col_key = type(programs[0])(prog)
                  except (ValueError, TypeError):
                      col_key = prog
                  col[col_key] = ";".join(str(g) for g in top5)
              # Fix #4: Typo FRD -> FDR
              df[f"Top 5 specific regulators (FDR<0.1) {condition}"] = col

      # Fix #1: Use len(programs) instead of k
      df['Total Enriched Enhancer Motifs'] = [''] * len(programs)
      df['Total Enriched Promoter Motifs'] = [''] * len(programs)

      # create top gene summary col
      if df_Program_loading is not None:
          df['top10_loaded_genes'] = [';'.join(df_Program_loading.loc[str(i)][:10]) for i in programs]
          df['top30_loaded_genes'] = [';'.join(df_Program_loading.loc[str(i)][:30]) for i in programs]

      # create explained variance summary col
      if df_Explained_Variance is not None:
          df['variance_explained'] = df_Explained_Variance
          
# make the program info in summary sheet
def get_program_info_Summary_cols(mdata, categorical_key = "sample"):

    # create cell info col summary
    df_cell = pd.DataFrame(data=mdata['cNMF'].X, index=mdata['cNMF'].obs_names, columns=mdata['cNMF'].var_names)
    results = []

    # use actual program names from mdata var_names
    programs = [int(v) for v in mdata['cNMF'].var_names]

    # Loop through all programs using label-based access
    for prog in programs:
        df_cell_program = pd.DataFrame({
            "expression": df_cell[str(prog)],
            "cell_type": mdata['rna'].obs[categorical_key].values
        })

        df_mean = df_cell_program.groupby("cell_type")["expression"].mean()
        df_frac = df_cell_program.groupby("cell_type")["expression"].apply(
            lambda x: (x > x.mean()).mean()
        )

        # Store as row with program as index
        results.append({
            'program_name': prog,
            **{f'Mean program score {ct}': df_mean[ct] for ct in df_mean.index},
            **{f'Fra cells above mean program score {ct}': df_frac[ct] for ct in df_frac.index}
        })

    df_cell_info = pd.DataFrame(results).set_index('program_name')

    return df_cell_info

# make top terms for the summray sheet
def get_top_terms_Summary_cols(df_GO, df_Geneset, adjusted_pval_key='Adjusted P-value'):

    top_GO = None
    top_Geneset = None

    if df_GO is not None:
        df_GO_filtered = df_GO[df_GO[adjusted_pval_key] < 0.05]
        top_GO = df_GO_filtered.groupby('program_name').apply(
            lambda x: ';'.join(x.sort_values(adjusted_pval_key).index[:10])
        )
    if df_Geneset is not None:
        df_Geneset_filtered = df_Geneset[df_Geneset[adjusted_pval_key] < 0.05]
        top_Geneset = df_Geneset_filtered.groupby('program_name').apply(
                lambda x: ';'.join(x.sort_values(adjusted_pval_key).index[:10])
            )

    top_terms = pd.DataFrame({
        'top10_enriched_genesets': top_Geneset,
        'top10_enriched_go_terms': top_GO
    })

    return top_terms

#-------------- helper methods summary--------------

# compile summry sheet
def Compile_Summary_sheet(mdata, df_GO, df_Geneset, df_Perturbation, df_Program_loading, df_Explained_Variance , specicicity_path = None, Sample = ["D0", "sample_D1","sample_D2","sample_D3"],
categorical_key = "sample",non_tagerting_key=None, effect_size='log2FC', adjusted_pval_key='Adjusted P-value'):

    print('complie summary sheet')

    # set program # from mdata (works even if df_GO is None)
    k = mdata['cNMF'].varm["loadings"].shape[0]
    programs = sorted(int(v) for v in mdata['cNMF'].var_names)

    df = pd.DataFrame({
    'manual_annotation_label': [''] * k,
    'manual_timepoint': [''] * k,
    'Notes': [''] * k,
    'Automatic Timepoint': [''] * k }, index=pd.Index(programs, name='program_name'))

    simple_Summary_cols(df, k, df_GO, df_Perturbation, df_Program_loading, df_Explained_Variance, specicicity_path = specicicity_path, Sample = Sample, non_tagerting_key=non_tagerting_key, effect_size=effect_size, adjusted_pval_key=adjusted_pval_key)
    df_cell_info_cols = get_program_info_Summary_cols(mdata,categorical_key)
    df_top_terms = get_top_terms_Summary_cols(df_GO, df_Geneset, adjusted_pval_key=adjusted_pval_key)

    # refill automatic time point
    col_names = [f'Mean program score {samp}' for samp in Sample]
    col_mapping = {f'Mean program score {samp}': samp for samp in Sample}      # Create mapping from column names to condition A, B, C
    df_mean = df_cell_info_cols[col_names]
    
    df['Automatic Timepoint'] = df_mean.idxmax(axis=1).map(col_mapping)

    merged_df = pd.merge(
        df,
        df_cell_info_cols,
        left_index=True,
        right_index=True,
        how='outer'
    )

    final_merged_df = pd.merge(
        merged_df,
        df_top_terms,
        left_index=True,
        right_index=True,
        how='outer'
    )

    # move top 10 loaded gene left of notes
    cols = final_merged_df.columns.tolist()                                                                                                                                                          
    cols.remove('top10_loaded_genes')
    i = cols.index('Notes')
    cols.insert(i, 'top10_loaded_genes')
    final_merged_df = final_merged_df[cols]

    return final_merged_df
   





