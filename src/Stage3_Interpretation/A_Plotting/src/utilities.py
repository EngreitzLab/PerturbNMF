import muon as mu 
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import scanpy as sc
import anndata as ad
from pathlib import Path
import mygene
import os
import shutil
import subprocess
from PyPDF2 import PdfMerger
import glob
import re                                                                                                                                                                                    
from svglib.svglib import svg2rlg                                                                                                                                                            
from reportlab.graphics import renderPDF

# replace EnsemblID by gene name given the dataframe with EnsemblID as the index
def convert_with_mygene(dataframe, species='human', index = True):

    mg = mygene.MyGeneInfo()
    
    if index:
        # Query multiple genes at once
        result = mg.querymany(dataframe.index, 
                            scopes='ensembl.gene', 
                            fields='symbol,name', 
                            species='human')
    else:
         # Query multiple genes at once
        result = mg.querymany(dataframe.columns, 
                            scopes='ensembl.gene', 
                            fields='symbol,name', 
                            species='human')

    
    # Create mapping dictionary
    mapping = {}
    for item in result:
        if 'symbol' in item and 'query' in item:
            mapping[item['query']] = item['symbol']

        elif 'query' in item:
            mapping[item['query']] = item['query']  # Keep original if no symbol

    if index: 
        new_dataframe = dataframe.rename(index=mapping)
    else:
        new_dataframe = dataframe.rename(columns=mapping)
    
    return new_dataframe



# same, input is adata
def convert_adata_with_mygene(adata, species='human'):

    mg = mygene.MyGeneInfo()
    gene_list = adata.var_names.tolist()
    annotations = mg.querymany(gene_list, scopes='ensembl.gene', 
                            fields='symbol', species='human')

    # Process the results to create mapping
    gene_dict = {}
    for item in annotations:
        if 'symbol' in item and 'query' in item:
            gene_dict[item['query']] = item['symbol']

        elif 'query' in item:
            gene_dict[item['query']] = item['query']  # Keep original if no symbol

    adata_new = adata.copy()


    adata_new.var['gene_name'] = [gene_dict.get(x, x) for x in adata_new.var_names]
    adata_new.var_names = adata_new.var['gene_name']

    return adata_new



# given a tsv dictionary, convert EnsemblID to gene name for adata 
def rename_adata_gene_dictionary(adata, dictionary_file_path):

    adata_new = adata.copy()

    # Convert mapping result to list before assignment
    df = pd.read_csv(dictionary_file_path, sep='\t', low_memory=False)
    ensemble_to_gene = dict(zip(df['ensembl_id'], df['gene']))
    new_names = [ensemble_to_gene.get(x, x) for x in adata.var_names]
    
    adata_new.var_names = (new_names)

    return adata_new



# given a tsv dictionary, convert EnsemblID to gene name for list
def rename_list_gene_dictionary(list_input, dictionary_file_path):

    # Convert mapping result to list before assignment
    df = pd.read_csv(dictionary_file_path, sep='\t', low_memory=False)
    ensemble_to_gene = dict(zip(df['ensembl_id'], df['gene']))
    new_names = [ensemble_to_gene.get(x, x) for x in list_input]
    
    return new_names



# read cNMF programs
def read_npz(path):

    # Load the NPZ file with pickle enabled
    npz_data = np.load(path, allow_pickle=True)

    # Reconstruct the DataFrame
    df = pd.DataFrame(
        data=npz_data['data'],
        index=npz_data['index'],
        columns=npz_data['columns']
    )
     
    return df



# Filter out the merged output itself and any 0-byte PDFs (silent failures from upstream workers).
# Both PyPDF2 and pdfunite choke on 0-byte inputs.
def _filter_mergeable_pdfs(pdf_files, output_path):
    kept, skipped_empty = [], []
    for p in pdf_files:
        if os.path.abspath(p) == os.path.abspath(output_path):
            continue
        if os.path.getsize(p) == 0:
            skipped_empty.append(p)
            continue
        kept.append(p)
    if skipped_empty:
        print(f"Skipping {len(skipped_empty)} empty (0-byte) PDF(s): {skipped_empty[:5]}{'...' if len(skipped_empty) > 5 else ''}")
    return kept


def _merge_pdfs(pdf_files, output_path):
    """Merge pdf_files -> output_path. Uses pdfunite (Poppler) when on PATH, falls back to PyPDF2.
    PyPDF2's PdfMerger hangs on thousands of inputs; pdfunite handles ~2k PDFs in seconds."""
    if not pdf_files:
        print("No PDFs to merge.")
        return

    pdfunite = shutil.which("pdfunite")
    if pdfunite is not None:
        try:
            subprocess.run([pdfunite, *pdf_files, output_path], check=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"pdfunite failed ({e}); falling back to PyPDF2.")

    merger = PdfMerger()
    for pdf_file in pdf_files:
        try:
            merger.append(pdf_file)
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    try:
        with open(output_path, 'wb') as output_file:
            merger.write(output_file)
    except Exception as e:
        print(f"Error saving merged PDF: {str(e)}")
    finally:
        merger.close()


# merge all PDFs into one, save pdf in the same folder_path
def merge_pdfs_in_folder(folder_path, output_filename="merged_perturbed_gene_QC.pdf"):
    output_path = os.path.join(folder_path, output_filename)

    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    pdf_files.sort(key=_natural_sort_key)
    print(f"Found {len(pdf_files)} PDF files")

    pdf_files = _filter_mergeable_pdfs(pdf_files, output_path)
    _merge_pdfs(pdf_files, output_path)



# merge all svgs to pdf
def merge_svgs_to_pdf(folder_path, output_filename="merged_perturbed_gene_QC.pdf"):

    svg_files = glob.glob(os.path.join(folder_path, "*.svg"))
    svg_files.sort(key=_natural_sort_key)
    print(f"Found {len(svg_files)} SVG files")

    output_path = os.path.join(folder_path, output_filename)
    temp_pdfs = []

    for svg_file in svg_files:
        try:
            drawing = svg2rlg(svg_file)
            temp_pdf = svg_file.replace('.svg', '_temp.pdf')
            renderPDF.drawToFile(drawing, temp_pdf)
            temp_pdfs.append(temp_pdf)
        except Exception as e:
            print(f"Error processing {svg_file}: {str(e)}")

    temp_pdfs = _filter_mergeable_pdfs(temp_pdfs, output_path)
    _merge_pdfs(temp_pdfs, output_path)

    for temp_pdf in temp_pdfs:
        try:
            os.remove(temp_pdf)
        except OSError:
            pass

    print(f"PDF created with {len(temp_pdfs)} pages: {output_path}")



def _natural_sort_key(filepath):
    """Sort key that orders numbers numerically (2 before 100) instead of lexicographically."""
    basename = os.path.basename(filepath)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', basename)]


def ensure_umap(mdata, data_key, prog_key, n_top_genes=2000, n_comps=50):
    """Compute PCA/UMAP from top-variance genes and write to both modalities; no-op if X_umap already exists in both."""
    has_prog = 'X_umap' in mdata[prog_key].obsm
    has_data = 'X_umap' in mdata[data_key].obsm
    if has_prog and has_data:
        return
    # If exactly one modality already has UMAP and the modalities share n_obs, mirror it
    # to the other instead of recomputing. This handles h5mu files where UMAP was stored
    # only under the cNMF (or only under the rna) modality.
    if has_prog ^ has_data:
        src_key, dst_key = (prog_key, data_key) if has_prog else (data_key, prog_key)
        if mdata[src_key].n_obs == mdata[dst_key].n_obs:
            mdata[dst_key].obsm['X_umap'] = mdata[src_key].obsm['X_umap']
            if 'X_pca' in mdata[src_key].obsm:
                mdata[dst_key].obsm['X_pca'] = mdata[src_key].obsm['X_pca']
            return
    # Compute variances on the live matrix (no copy) so a 1M×30K matrix doesn't get duplicated.
    X = mdata[data_key].X
    variances = np.array(X.power(2).mean(axis=0) - np.power(X.mean(axis=0), 2)).flatten() \
        if hasattr(X, 'power') else X.var(axis=0)
    top_idx = np.argsort(variances)[-n_top_genes:]
    # Slice first, copy second: the copy is now ~n_top_genes columns instead of the full matrix.
    adata_tmp = mdata[data_key][:, top_idx].copy()
    sc.tl.pca(adata_tmp, n_comps=n_comps)
    sc.pp.neighbors(adata_tmp)
    sc.tl.umap(adata_tmp)
    mdata[prog_key].obsm['X_pca'] = adata_tmp.obsm['X_pca']
    mdata[prog_key].obsm['X_umap'] = adata_tmp.obsm['X_umap']
    mdata[data_key].obsm['X_pca'] = adata_tmp.obsm['X_pca']
    mdata[data_key].obsm['X_umap'] = adata_tmp.obsm['X_umap']


def check_normalized(adata, key):
    """Raise if adata.X looks like raw integer counts instead of normalized expression."""
    import scipy.sparse as sp
    X = adata.X
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if vals.size == 0:
        return
    if vals.size > 100_000:
        vals = np.random.default_rng(0).choice(vals, 100_000, replace=False)
    if np.all(np.mod(vals, 1) == 0) and float(vals.max()) > 30:
        raise ValueError(
            f"mdata['{key}'].X appears to contain raw integer counts "
            f"(max={float(vals.max()):g}). This script requires library-size-"
            "normalized (e.g. TPM/CPM) or log-normalized expression. Normalize "
            "before running, e.g.:\n"
            "    import scanpy as sc\n"
            "    sc.pp.normalize_total(adata, target_sum=1e4)\n"
            "    sc.pp.log1p(adata)\n"
            "and save the normalized matrix into mdata['rna'].X."
        )