import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib import gridspec
import scanpy as sc
from pathlib import Path
import anndata
import muon 
from torch_cnmf import cNMF
import os
import sys
import mygene
import shutil
from pathlib import Path
from tqdm.auto import tqdm

def run_cnmf_consensus(cnmf_obj=None, output_dir=None, name=None, 
                       components=[7,8,9,10], density_thresholds=[0.01, 0.05, 2.0]):
    """
    Run cNMF consensus analysis for multiple k values and density thresholds.
    
    Args:
        cnmf_obj: Existing cNMF object, or None to create a new one
        output_dir: Directory to save results
        name: Name prefix for output files
        components: List of k values (number of components) to test
        density_thresholds: List of density threshold values for consensus
    """
    if cnmf_obj is None:
        cnmf_obj = cNMF(output_dir=output_dir, name=name)

    for k in tqdm(components, desc='Running cNMF'):
        for thresh in density_thresholds:
            cnmf_obj.consensus(k=k, density_threshold=thresh, show_clustering=True)


def filter_noncoding_genes(counts_fn, inference_dir, gene_names_key='symbol', ensembl_prefix='ENSG'):
    """Filter out non-coding genes (whose name starts with `ensembl_prefix`) and write a copy.

    Returns the path to the filtered .h5ad. Caller should rebind whatever variable
    holds the input counts path to this return value before downstream steps.

    Args:
        counts_fn: Path to input .h5ad.
        inference_dir: Directory to write the filtered .h5ad into.
        gene_names_key: Column in adata.var that holds the gene symbol/ID used for filtering.
        ensembl_prefix: Strings starting with this prefix are dropped.

    Returns:
        Path to the filtered .h5ad written under `inference_dir`.
    """
    adata = sc.read(counts_fn)
    mask = ~adata.var[gene_names_key].str.startswith(ensembl_prefix)
    adata = adata[:, mask].copy()
    filtered_path = f'{inference_dir}/adata_without_noncoding.h5ad'
    adata.write(filtered_path)
    return filtered_path


def _strip_ensembl_version(ids):
    """Return Ensembl IDs with any trailing `.<version>` removed (e.g. ENSG0001.12 -> ENSG0001)."""
    return pd.Index(ids).astype(str).str.split('.').str[0]


def load_protein_coding_ids(gtf_path, gene_type='protein_coding'):
    """Parse a (optionally gzipped) GENCODE/Ensembl GTF and return the set of
    version-stripped gene IDs whose `gene` feature line has the requested `gene_type`.

    Args:
        gtf_path: Path to a GTF/GTF.gz file.
        gene_type: gene_type attribute value to keep (default 'protein_coding').

    Returns:
        A set of version-stripped Ensembl gene IDs (e.g. {'ENSG00000121410', ...}).
    """
    import gzip
    import re

    gene_id_re = re.compile(r'gene_id "([^"]+)"')
    gene_type_re = re.compile(r'gene_type "([^"]+)"')

    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    keep = set()
    with opener(gtf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            cols = line.split('\t', 8)
            if len(cols) < 9 or cols[2] != 'gene':
                continue
            attrs = cols[8]
            gt = gene_type_re.search(attrs)
            if gt is None or gt.group(1) != gene_type:
                continue
            gid = gene_id_re.search(attrs)
            if gid is not None:
                keep.add(gid.group(1).split('.')[0])
    return keep


def filter_noncoding_genes_gtf(counts_fn, inference_dir, gtf_path,
                               gene_id_key='gene_id', gene_type='protein_coding'):
    """Keep only genes whose Ensembl ID is annotated as `gene_type` in `gtf_path`; drop the rest.

    Unlike `filter_noncoding_genes` (a prefix heuristic on the symbol column), this
    uses the GTF as ground truth: any gene whose Ensembl ID is not listed as a
    `protein_coding` `gene` feature in the GTF is removed. Ensembl version suffixes
    (`.<n>`) are stripped on both sides before matching.

    Returns the path to the filtered .h5ad. Caller should rebind whatever variable
    holds the input counts path to this return value before downstream steps.

    Args:
        counts_fn: Path to input .h5ad.
        inference_dir: Directory to write the filtered .h5ad into.
        gtf_path: Path to a GTF/GTF.gz file (GENCODE/Ensembl).
        gene_id_key: Column in adata.var holding the Ensembl gene ID. If absent,
            falls back to adata.var_names.
        gene_type: gene_type attribute value to keep (default 'protein_coding').

    Returns:
        Path to the filtered .h5ad written under `inference_dir`.
    """
    coding_ids = load_protein_coding_ids(gtf_path, gene_type=gene_type)
    if len(coding_ids) == 0:
        raise ValueError(
            f"No '{gene_type}' gene features found in GTF: {gtf_path}. "
            "Check the file path and that it is a GENCODE/Ensembl GTF with gene_type attributes."
        )

    adata = sc.read(counts_fn)
    if gene_id_key in adata.var.columns:
        raw_ids = adata.var[gene_id_key]
    else:
        print(f"WARNING: '{gene_id_key}' not in adata.var columns; falling back to var_names.")
        raw_ids = adata.var_names
    base_ids = _strip_ensembl_version(raw_ids)
    mask = np.asarray(base_ids.isin(coding_ids))

    n_keep = int(mask.sum())
    if n_keep == 0:
        raise ValueError(
            f"0 of {adata.n_vars} genes matched '{gene_type}' IDs from the GTF. "
            f"Check that adata.var['{gene_id_key}'] holds Ensembl gene IDs (e.g. ENSG...)."
        )
    print(f"GTF filter: keeping {n_keep}/{adata.n_vars} genes "
          f"({gene_type} in {os.path.basename(str(gtf_path))}).")

    adata = adata[:, mask].copy()
    filtered_path = f'{inference_dir}/adata_without_noncoding.h5ad'
    adata.write(filtered_path)
    return filtered_path


def load_gene_id_to_name(gtf_path):
    """Parse a (optionally gzipped) GENCODE/Ensembl GTF and return a mapping from
    version-stripped gene_id to gene_name (from the `gene` feature lines).

    Args:
        gtf_path: Path to a GTF/GTF.gz file.

    Returns:
        dict {version-stripped Ensembl gene ID -> gene_name}
        (e.g. {'ENSG00000121410': 'A1BG', ...}).
    """
    import gzip
    import re

    gene_id_re = re.compile(r'gene_id "([^"]+)"')
    gene_name_re = re.compile(r'gene_name "([^"]+)"')

    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    id2name = {}
    with opener(gtf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            cols = line.split('\t', 8)
            if len(cols) < 9 or cols[2] != 'gene':
                continue
            attrs = cols[8]
            gid = gene_id_re.search(attrs)
            name = gene_name_re.search(attrs)
            if gid is not None and name is not None:
                id2name[gid.group(1).split('.')[0]] = name.group(1)
    return id2name


def add_gene_names_from_gtf(counts_fn, inference_dir, gtf_path,
                            gene_id_key='gene_id', gene_names_key='symbol',
                            keep_unmatched=True):
    """Add/overwrite adata.var[gene_names_key] with gene symbols looked up from the
    GTF by (version-stripped) Ensembl ID, then write a copy.

    Returns the path to the annotated .h5ad. Caller should rebind whatever variable
    holds the input counts path to this return value before downstream steps.

    Args:
        counts_fn: Path to input .h5ad.
        inference_dir: Directory to write the annotated .h5ad into.
        gtf_path: Path to a GTF/GTF.gz file (GENCODE/Ensembl).
        gene_id_key: Column in adata.var holding the Ensembl gene ID. If absent,
            falls back to adata.var_names.
        gene_names_key: Column in adata.var to write the resolved gene symbols into
            (created if absent, overwritten if present).
        keep_unmatched: If True, IDs with no GTF match keep their original Ensembl ID
            as the name; if False, unmatched names are set to empty string.

    Returns:
        Path to the annotated .h5ad written under `inference_dir`.
    """
    id2name = load_gene_id_to_name(gtf_path)
    if len(id2name) == 0:
        raise ValueError(
            f"No gene_id->gene_name pairs found in GTF: {gtf_path}. "
            "Check the file path and that it is a GENCODE/Ensembl GTF with gene_name attributes."
        )

    adata = sc.read(counts_fn)
    if gene_id_key in adata.var.columns:
        raw_ids = adata.var[gene_id_key]
    else:
        print(f"WARNING: '{gene_id_key}' not in adata.var columns; falling back to var_names.")
        raw_ids = adata.var_names
    base_ids = _strip_ensembl_version(raw_ids)

    mapped = base_ids.map(id2name)
    n_matched = int(mapped.notna().sum())
    if n_matched == 0:
        raise ValueError(
            f"0 of {adata.n_vars} genes matched a gene_name from the GTF. "
            f"Check that adata.var['{gene_id_key}'] holds Ensembl gene IDs (e.g. ENSG...)."
        )

    fallback = pd.Index(base_ids).astype(str) if keep_unmatched else ""
    names = mapped.where(mapped.notna(), fallback)
    adata.var[gene_names_key] = np.asarray(names)

    print(f"GTF gene names: matched {n_matched}/{adata.n_vars} genes; "
          f"wrote adata.var['{gene_names_key}'] from {os.path.basename(str(gtf_path))}.")

    annotated_path = f'{inference_dir}/adata_with_gene_names.h5ad'
    adata.write(annotated_path)
    return annotated_path


def compile_results(output_directory, run_name, sel_threshs = [2.0], components = [30, 50, 60, 80, 100, 200, 250, 300],
 guide_names_key = "guide_names", guide_targets_key = "guide_targets", categorical_key= 'batch', guide_assignment_key ="guide_assignment",
 gene_names_key = None ):
    """
    Compile cNMF results into correct format for downstream evaluation pipeline.
    
    Processes cNMF output files and reorganizes them into standardized formats
    including scores, loadings, and AnnData objects for further analysis.
    
    Args:
        output_directory: Base directory containing cNMF results
        run_name: Name of the cNMF run
        sel_thresh: List of density threshold values to process
        components: List of k values (number of components) to process
    """


    adata_ = anndata.read_h5ad('{output_directory}/{run_name}/cnmf_tmp/{run_name}.tpm.h5ad'.format(
                                                                                    output_directory=output_directory,
                                                                                    run_name = run_name))
       
    for i in sel_threshs:
        for k in components:

            scores = pd.read_csv('{output_directory}/{run_name}/{run_name}.usages.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                            output_directory=output_directory,
                                                                                            run_name = run_name,
                                                                                            k=k,
                                                                                            sel_thresh = str(i).replace('.','_')),
                                                                                            sep='\t', index_col=0)

            loadings = pd.read_csv('{output_directory}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                            output_directory=output_directory,
                                                                                            run_name = run_name,
                                                                                            k=k,
                                                                                            sel_thresh = str(i).replace('.','_')),
                                                                                            sep='\t', index_col=0)
            

            os.makedirs((f'{output_directory}/{run_name}/loading'), exist_ok=True)


            scores.to_csv('{output_directory}/{run_name}/loading/cNMF_scores_{k}_{sel_thresh}.txt'.format(
                                                                                            output_directory=output_directory,
                                                                                            run_name = run_name,
                                                                                            k=k,
                                                                                            sel_thresh = str(i).replace('.','_')), sep='\t')
            loadings.T.to_csv('{output_directory}/{run_name}/loading/cNMF_loadings_{k}_{sel_thresh}.txt'.format(
                                                                                            output_directory=output_directory,
                                                                                            run_name = run_name,
                                                                                            k=k,
                                                                                            sel_thresh = str(i).replace('.','_')), sep='\t')

            adata_.var_names_make_unique()
            adata_.obs_names_make_unique()

            prog_data = anndata.AnnData(X=scores.values, obs=adata_.obs)
            prog_data.var_names = scores.columns.values
            prog_data.varm['loadings'] = loadings.values
            # Use gene_names_key column if provided, otherwise fall back to var_names / loadings columns
            if gene_names_key is not None and gene_names_key in adata_.var.columns:
                gene_names = adata_.var[gene_names_key].values
            else:
                gene_names = loadings.columns.values

            prog_data.uns['var_names'] = gene_names


            # Make adata
            os.makedirs((f'{output_directory}/{run_name}/prog_data'), exist_ok=True)
            prog_data.write('{output_directory}/{run_name}/prog_data/NMF_{k}_{sel_thresh}.h5ad'.format(
                                                                                    output_directory=output_directory,
                                                                                    run_name = run_name,
                                                                                    k=k,
                                                                                    sel_thresh = str(i).replace('.','_')))
            # Make mdata
            mdata = muon.MuData({'rna': adata_, 'cNMF': prog_data})
            if guide_names_key and guide_names_key in adata_.uns:
                mdata['cNMF'].uns[guide_names_key] = adata_.uns[guide_names_key]
            if guide_targets_key and guide_targets_key in adata_.uns:
                mdata['cNMF'].uns[guide_targets_key] = adata_.uns[guide_targets_key]
            if categorical_key and categorical_key in adata_.obs.columns:
                mdata['cNMF'].obs[categorical_key] = adata_.obs[categorical_key]
            if guide_assignment_key and guide_assignment_key in adata_.obsm:
                mdata['cNMF'].obsm[guide_assignment_key] = adata_.obsm[guide_assignment_key]
            if 'X_pca' in adata_.obsm:
                mdata['cNMF'].obsm['X_pca'] = adata_.obsm['X_pca']
            if 'X_umap' in adata_.obsm:
                mdata['cNMF'].obsm['X_umap'] = adata_.obsm['X_umap']

            if gene_names_key is not None and gene_names_key in adata_.var.columns:
                mdata['rna'].var['var_names'] = adata_.var[gene_names_key].values
            else:
                mdata['rna'].var['var_names'] = adata_.var_names


            os.makedirs((f'{output_directory}/{run_name}/adata'), exist_ok=True)
            mdata.write('{output_directory}/{run_name}/adata/cNMF_{k}_{sel_thresh}.h5mu'.format(
                                                                                    output_directory=output_directory,
                                                                                    run_name = run_name,
                                                                                    k=k,
                                                                                    sel_thresh = str(i).replace('.','_')))
    


def get_top_indices_fast(df, gene_num = 300):
    """
    Given a DataFrame from cNMF, return top N genes for each program.
    
    Args:
        df: DataFrame with programs as rows and genes as columns
        gene_num: Number of top genes to return per program
    
    Returns:
        DataFrame with programs as rows and top genes as columns
    """
 
    # Get column names
    col_names = df.columns.values
    
    # Use argsort to get indices of top 300 values per row
    # argsort sorts in ascending order, so we use [:, -300:] and reverse
    top_indices = np.argsort(df.values, axis=1)[:, -gene_num:][:, ::-1]
    
    # Map indices to column names
    top_col_names = col_names[top_indices]
    
    # Create result DataFrame
    result_df = pd.DataFrame(
        top_col_names,
        index=df.index,
        columns=[f'top_{i+1}' for i in range(gene_num)]
    )

    result_df.index = [f'Program_{i}' for i in range(1,len(result_df)+1)]
    
    return result_df


def annotate_genes_to_excel(df, species = 'human', output_file='gene_annotations.xlsx'):
    """
    Annotate genes and export to Excel file.
    
    Takes a DataFrame with programs as rows and genes as columns,
    queries MyGene database for annotations, and exports results to Excel.
    
    Args:
        df: DataFrame with rows for each program, columns for genes
        species: Species for gene annotation ('human' or 'mouse')
        output_file: Path for output Excel file
    
    Returns:
        Dictionary of annotations for each program
    """
    
    # Initialize MyGene
    mg = mygene.MyGeneInfo()
    
    # Dictionary to store results for each column
    all_annotations = {}
    
    # Process each column
    for row_idx in df.index:
        # Get unique genes from column (remove NaN)
        genes = df.loc[row_idx].dropna().unique().tolist()
        
        if len(genes) == 0:
            print(f"Column '{row_idx}': No genes found")
            continue
        
        print(f"Annotating column '{row_idx}': {len(genes)} genes...")
        
        # Query MyGene for annotations
        results = mg.querymany(
            genes, 
            scopes='symbol,alias,ensembl.gene',  # Multiple search scopes
            fields='symbol,name,entrezgene,summary,type_of_gene',
            species=species,
            returnall=True
        )
        
        # Build lookup: for each query, keep the highest-scoring hit
        best_hit = {}
        for gene_info in results['out']:
            query = gene_info.get('query', '')
            if 'notfound' in gene_info and gene_info['notfound']:
                if query not in best_hit:
                    best_hit[query] = gene_info
            else:
                score = gene_info.get('_score', 0)
                if query not in best_hit or 'notfound' in best_hit[query] or score > best_hit[query].get('_score', 0):
                    best_hit[query] = gene_info

        # Process results in original gene order
        annotation_list = []
        for query_gene in genes:
            gene_info = best_hit.get(query_gene)
            if gene_info is None or ('notfound' in gene_info and gene_info['notfound']):
                annotation_list.append({
                    'Input_Gene': query_gene,
                    'Gene_Symbol': 'NOT FOUND',
                    'Gene_Name': 'NOT FOUND',
                    'Entrez_ID': '',
                    'Type': '',
                    'Summary': ''
                })
            else:
                annotation_list.append({
                    'Input_Gene': query_gene,
                    'Gene_Symbol': gene_info.get('symbol', query_gene),
                    'Gene_Name': gene_info.get('name', ''),
                    'Entrez_ID': gene_info.get('entrezgene', ''),
                    'Type': gene_info.get('type_of_gene', ''),
                    'Summary': gene_info.get('summary', '')
                })
        
        # Create DataFrame for this column
        all_annotations[row_idx] = pd.DataFrame(annotation_list)

    # Export to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for row_idx, annotations_df in all_annotations.items():
            sheet_name = str(row_idx)
            # Truncate sheet name if too long (Excel limit is 31 chars)
            sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
            annotations_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return all_annotations


def rename_and_move_files_NMF(file_name_input, file_name_output, source_folder, destination_folder, len = 10):
    """
    Combine parallel inferred cNMF component results into one run.
    
    Renames and moves NMF result files from parallel runs to a unified location.
    
    Args:
        file_name_input: Input filename pattern
        file_name_output: Output filename pattern
        source_folder: Source directory containing parallel results
        destination_folder: Destination directory for combined results
        len: Number of parallel runs to process
    
    Returns:
        List of tuples with (source_filename, new_filename) for processed files
    """

    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    # Process files for i 
    for i in range(len):
        source_filename = f"{file_name_input}_{i}.df.npz"
        source_path = os.path.join(source_folder, source_filename)
        
        # Check if source file exists
        if os.path.exists(source_path):
            
            # Create new filename
            new_filename = f"{file_name_output}_{i}.df.npz"
            destination_path = os.path.join(destination_folder, new_filename)
            
            try:
                # Copy and rename file to destination
                shutil.copy2(source_path, destination_path)
                print(f"Successfully processed: {source_filename} -> {new_filename}")
                processed_files.append((source_filename, new_filename))
                
            except Exception as e:
                print(f"Error processing {source_filename}: {e}")
        else:
            print(f"File not found: {source_filename}")
    
    return processed_files


def rename_all_NMF(file_name_input, file_name_output, source_folder, destination_folder, len = 10, components = [30, 50, 60, 80, 100, 200, 250, 300]):
    """
    Combine all K values from parallel runs.
    
    Processes results from multiple parallel cNMF runs across different k values.
    
    Args:
        file_name_input: Input filename pattern
        file_name_output: Output filename pattern
        source_folder: Base source directory pattern
        destination_folder: Destination directory for combined results
        len: Number of parallel runs per k value
        components: List of k values to process
    """

    for k in components:

        file_name_input_new = f"{file_name_input}.spectra.k_{k}.iter"
        file_name_output_new = f"{file_name_output}.spectra.k_{k}.iter"

        source_folder_new = f"{source_folder}_{k}/Inference/cnmf_tmp"

        rename_and_move_files_NMF(file_name_input_new, file_name_output_new, source_folder_new, destination_folder, len=len)
