import os
import sys
import yaml
import logging
import mudata as mu
import pandas as pd
import scanpy as sc
import mygene
import numpy as np

def check_data_format(adata, guide_names_key = "guide_names", guide_targets_key = "guide_targets", categorical_key= 'batch',
guide_assignment_key = 'guide_assignment'):
    """
    Validate that the AnnData object has the correct format for cNMF analysis.
    
    Checks for:
    - Required categorical variable in obs
    - guide_names and guide_targets in uns
    - PCA and UMAP embeddings in obsm
    - guide_assignment matrix in obsm (converts from sparse to dense if needed)
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to validate
    guide_names_key : str, optional
        Key for guide names in adata.uns (default: 'guide_names')
    guide_targets_key : str, optional
        Key for guide targets in adata.uns (default: 'guide_targets')
    categorical_key : str, optional
        Key for categorical variable in adata.obs (default: 'batch')
    
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    
    Notes
    -----
    Prints warnings for each validation failure
    """
    is_valid = True
    
    
    # Check if the specified categorical variable exists in obs
    if categorical_key not in adata.obs:
        print(f"WARNING: Not found in adata.obs['{categorical_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.obs['{categorical_key}']\n")


    # Check if guide names metadata exists in uns
    if guide_names_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_names_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_names_key}']\n")
    
    # Check if guide targets metadata exists in uns
    if guide_targets_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_targets_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_targets_key}']\n")


    # Check if PCA embeddings exist in obsm
    if 'X_pca' not in adata.obsm:
        print(f"WARNING: Not found adata.obsm['X_pca'] \n")
        is_valid = False
    else:
        print(f"Found adata.obsm['X_pca']\n")

    # Check if UMAP embeddings exist in obsm
    if 'X_umap' not in adata.obsm:
        print(f"WARNING: Not found adata.obsm['X_umap'] \n")
        is_valid = False
    else:
        print(f"Found adata.obsm['X_umap']\n")
    
    # Check if guide assignment matrix exists in obsm
    if 'guide_assignment' not in adata.obsm:
        print(f"WARNING: Not found adata.obsm['guide_assignment'] \n")
        is_valid = False
    else:
        guide_assignment = adata.obsm[guide_assignment_key]
        print(f"Found adata.obsm['{guide_assignment_key}']\n")
        
        # Ensure guide_assignment is in dense format (required for downstream analysis)
        try:
            import scipy.sparse as sp
            is_sparse = sp.issparse(guide_assignment)
            
            if is_sparse:
                print(f"WARNING: '{guide_assignment_key}' is sparse. Converting to dense array...")
                dense_array = guide_assignment.toarray()
                adata.obsm[guide_assignment_key] = dense_array
                print(f"'{guide_assignment_key}' converted to dense array (shape: {dense_array.shape}) \n")
            else:
                print(f"'{guide_assignment_key}' is already dense (shape: {guide_assignment.shape}) \n")
        except Exception as e:
            print(f"WARNING: Error checking '{guide_assignment_key}' sparsity: {e} \n")
            is_valid = False

    return is_valid

def check_guide_names(adata, guide_names_key = "guide_names", guide_targets_key = "guide_targets", categorical_key= 'batch', 
reference_gtf_path=None, guide_annotation_path = None, guide_assignment_key = 'guide_assignment', check_var_names_align_with_guide_targets= True):
    """
    Validate gene names consistency across RNA modality and cNMF guide names,
    with optional reference GTF validation.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing 'rna' and 'cNMF' modalities
    prog_key : str
        index for the gene program anndata object (mdata[prog_key]) in the mudata object.   
    data_key : str
        index for the gene expression anndata object (mdata[prog_key]) in the mudata object.   
    categorical_key : str
        index for the gene name anndata object (mdata[data_key]) in the mudata object.
    reference_gtf_path : str, optional
        Path to reference GTF file (e.g., IGVF GENCODE v43 GTF).
        If provided, checks that gene names in mdata['rna'].var match the GTF.
        GTF should have a gene_id or gene_name column.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'is_valid' (bool): True if all checks pass
        - 'rna_vs_guide_mismatches' (list): Gene names in RNA but not in guide_names
        - 'missing_from_rna' (list): Gene names in guide_names but not in RNA
        - 'gtf_validation' (dict or None): GTF validation results if reference provided
    
    Raises
    ------
    Prints warnings for each validation failure
    """
    results = {
        'is_valid': True,
        'missing_from_rna': [],
        'gtf_validation': None
     }

    results['is_valid'] = check_data_format(adata,  guide_names_key = guide_names_key, guide_targets_key = guide_targets_key, 
    categorical_key= categorical_key, guide_assignment_key = guide_assignment_key)
    
    rna_gene_names = set(adata.var_names)

    guide_targets = adata.uns[guide_targets_key]
    guide_names = adata.uns[guide_names_key]

    
    # Convert guide data to sets, handling different pandas/numpy data types
    if isinstance(guide_targets, (list, tuple)):
        guide_gene_targets = set(guide_targets)
        guide_gene_names = set(guide_names)
    elif isinstance(guide_targets, pd.Index):
        guide_gene_targets = set(guide_targets.values)
        guide_gene_names = set(guide_names.values)
    elif isinstance(guide_targets, pd.Series):
        guide_gene_targets = set(guide_targets.values)
        guide_gene_names = set(guide_names.values)
    else:
        guide_gene_targets = set(guide_targets)
        guide_gene_names = set(guide_names)

    # check equal length
    if len(guide_targets)!=len(guide_names) and len(guide_gene_targets)!=((adata.obsm[guide_assignment_key]).shape[1]):
         print(f"WARNING: guide_targets and guide_names and guide_assignment col should be equal length but not in here\n")
         results['is_valid'] = False

    if check_var_names_align_with_guide_targets:

        # Find guide targets that are missing from RNA gene names
        guide_not_in_rna = guide_gene_targets - rna_gene_names
        print(f"Found {len(guide_gene_targets)} genes in gene_targets")

        if guide_not_in_rna:
            print(f"WARNING: {len(guide_not_in_rna)}/{len(guide_gene_targets)} gene names in guide_targets but NOT in adata.var. \n \
                This might be caused by mismatch between gene symbol vs Ensembl ID or some perturbed genes are not expressed in the dataset. \
                Might cause issue in evalutation or plotting if naming conventions are different.")
            print(f"Examples: {list(guide_not_in_rna)[:5]}\n")
            results['missing_from_rna'] = sorted(list(guide_not_in_rna))
        else:
            print("All guide_targets found in RNA gene names\n")
    
    # Optional: validate against reference GTF file
    if reference_gtf_path is None:
        print("No reference GTF provided (optional check skipped)")
        print("To validate against a reference, provide reference_gtf_path parameter")
        results['gtf_validation'] = None
    else:
        gtf_validation = _validate_against_reference_gtf(
            rna_gene_names, 
            reference_gtf_path
        )
        results['gtf_validation'] = gtf_validation
        
        if not gtf_validation['is_valid']:
           results['is_valid'] = False
    
    # Optional: validate against guide annotation file
    if guide_annotation_path is None:
        print("\nNo reference guide annotation provided (optional check skipped)")
        print("To validate against a reference, provide guide annotation parameter")
        results['guide_annotation'] = None
    else:
        guide_annotation_df = pd.read_csv(guide_annotation_path, sep='\t', index_col = 0)

        # Check if required columns exist in guide annotation file
        if guide_names_key not in guide_annotation_df.columns:
            print(f"{guide_names_key} not in guide annotation file")
            results['is_valid'] = False
            return results
        else:
            annotation_guide_names = set(guide_annotation_df[guide_names_key].values)

        # Check if guide targets column exists in annotation file
        if guide_targets_key not in guide_annotation_df.columns:
            results['is_valid'] = False
            print(f"{guide_targets_key} not in guide annotation file")
            return results
        else:
            annotation_guide_targets = set(guide_annotation_df[guide_targets_key].values)

        #check targeting columns exits
        if "targeting" not in guide_annotation_df.columns:
            results['is_valid'] = False
            print(f"'targeting' not in guide annotation file")

        # Compare data guides with annotation guides
        guide_targets_not_in_annotation = guide_gene_targets - annotation_guide_targets
        guide_names_not_in_annotation = guide_gene_names - annotation_guide_names

        if guide_targets_not_in_annotation or guide_names_not_in_annotation:
            if guide_targets_not_in_annotation:
                print(f"\n WARNING: {len(guide_targets_not_in_annotation)}/{len(guide_gene_targets)} guide targets in data but NOT in annotation.")
                print(f"Examples: {list(guide_targets_not_in_annotation)[:5]}\n")
            
            if guide_names_not_in_annotation:
                print(f"\n WARNING: {len(guide_names_not_in_annotation)}/{len(guide_gene_names)} guide names in data but NOT in annotation.")
                print(f"Examples: {list(guide_names_not_in_annotation)[:5]}\n")

            results['is_valid'] = False
        else:
            print("All guide_targets and guide_names match guide annotation\n")


    return results

def _validate_against_reference_gtf(rna_gene_names, gtf_path):
    """
    Validate RNA gene names against a reference GTF file.
    
    Extracts gene symbols from GTF file and compares with RNA gene names.
    Helps identify mismatches between Ensembl IDs and gene symbols.
    
    Parameters
    ----------
    rna_gene_names : set
        Set of gene names from RNA data
    gtf_path : str
        Path to GTF file containing gene annotations
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid' (bool): True if validation succeeded
        - 'gtf_genes' (set): Gene symbols extracted from GTF
        - 'not_in_gtf' (list): RNA genes not found in GTF
    """
    validation = {
        'is_valid': True,
        'gtf_genes': set(),
        'not_in_gtf': [],
     }
    
    try:
        # Read GTF file with standard 9-column format
        gtf_df = pd.read_csv(
            gtf_path, 
            sep='\t', 
            comment='#', 
            header=None,
            dtype={0: str, 1: str, 2: str, 3: int, 4: int, 5: str, 6: str, 7: str, 8: str}
        )
        
        # Extract gene symbols from GTF attributes column using regex
        extracted = gtf_df[8].str.extract(r'gene_name "([^"]+)"')[0]

        if extracted.notna().sum() == 0:
            print("gene_name not found in any rows \n")
        else:
            print(f"Successfully extracted gene_name from {extracted.notna().sum()} rows")
            extracted = set(extracted)
            validation['gtf_genes'] = extracted
                
        
        # Find RNA gene names that are not in the GTF reference
        rna_not_in_gtf = rna_gene_names - extracted
        print(f"Found {len(rna_gene_names)} gene names in adata.var_name")

        
        if rna_not_in_gtf:
            print(f"WARNING: {len(rna_not_in_gtf)}/{len(rna_gene_names)} adata.var RNA NOT found in reference GTF. \
                \n This might be caused by the adata.var RNA being Ensembl ID instead of gene symbols.")
            print(f"Examples: {list(rna_not_in_gtf)[:5]} \n")
            validation['not_in_gtf'] = sorted(list(rna_not_in_gtf))
        else:
            print("All RNA genes found in reference GTF \n")
    

    except FileNotFoundError:
        print(f"ERROR: Reference GTF file not found at {gtf_path} \n")
        validation['is_valid'] = False
    except Exception as e:
        print(f"ERROR: Failed to read reference GTF: {e} \n")
        validation['is_valid'] = False
    
    return validation
      
def check_mdata_format(mdata, prog_key = 'cNMF', data_key = 'rna', guide_names_key = "guide_names", guide_targets_key = "guide_targets", 
categorical_key= 'batch', reference_gtf_path=None, guide_annotation_path = None, guide_assignment_key = 'guide_assignment'):
    """
    Validate gene names consistency in a MuData object containing both RNA and cNMF modalities.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing multiple modalities
    prog_key : str, optional
        Key for cNMF program modality (default: 'cNMF')
    data_key : str, optional
        Key for RNA expression modality (default: 'rna')
    guide_names_key : str, optional
        Key for guide names in uns (default: 'guide_names')
    guide_targets_key : str, optional
        Key for guide targets in uns (default: 'guide_targets')
    categorical_key : str, optional
        Key for categorical variable in obs (default: 'batch')
    reference_gtf_path : str, optional
        Path to reference GTF file for validation
    guide_annotation_path : str, optional
        Path to guide annotation file for validation
        
    Returns
    -------
    bool
        True if all validation checks pass
    """
    is_valid = True

    
    # Validate RNA expression modality exists
    if data_key not in mdata.mod:
        print(f"WARNING: {data_key} modality not found in mdata \n")
        is_valid = False
        return is_valid
    else:
        print(f"mdata[{data_key}] modality found\n")

    # Validate RNA data format
    rna_adata = mdata[data_key]
    is_valid= check_guide_names(rna_adata, guide_names_key = guide_names_key, guide_targets_key = guide_targets_key, 
    categorical_key= categorical_key, guide_assignment_key=guide_assignment_key, reference_gtf_path=reference_gtf_path,
    guide_annotation_path=guide_annotation_path,check_var_names_align_with_guide_targets = True)

    # Validate cNMF program modality exists
    if prog_key not in mdata.mod:
        print(f"WARNING: {prog_key} modality not found in mdata\n")
        is_valid = False
        return is_valid
    else:
        print(f"mdata[{prog_key}] modality found\n")

    # Validate cNMF program data format
    cnmf_adata = mdata[prog_key]
    is_valid = check_guide_names(cnmf_adata, guide_names_key = guide_names_key, guide_targets_key = guide_targets_key, 
    categorical_key= categorical_key,guide_assignment_key=guide_assignment_key, reference_gtf_path = None,
    guide_annotation_path=guide_annotation_path, check_var_names_align_with_guide_targets = False)

    # Check if PCA loadings exist in variable metadata (varm)
    if 'loadings' not in cnmf_adata.varm:
        print(f"WARNING: Not found adata.varm['loadings'] \n")
        is_valid = False
    else:
        print(f"Found adata.varm['loadings']\n")


    return is_valid

