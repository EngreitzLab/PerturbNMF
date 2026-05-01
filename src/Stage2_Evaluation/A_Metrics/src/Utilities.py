import os
import sys
import yaml
import logging
import mudata as mu
import pandas as pd
import torch_cnmf
import scanpy as sc
import mygene



def rename_gene_mygene(mdata):
    """
    Convert Ensembl gene IDs to gene symbols using the MyGene web service.
    
    Rename genes in cNMF modality using MyGene service to convert Ensembl IDs to gene symbols.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing cNMF modality with var_names to be renamed
    """

    mg = mygene.MyGeneInfo()
    gene_list = mdata['cNMF'].uns['var_names'].tolist()
    annotations = mg.querymany(gene_list, scopes='ensembl.gene', fields='symbol', species='human')

    # Process the results to create mapping
    gene_dict = {}
    
    for item in annotations:
        if 'symbol' in item:
            gene_dict[item['query']] = item['symbol']

    mdata['cNMF'].uns['var_names'] = [gene_dict.get(x, x) for x in mdata['cNMF'].uns['var_names']]

def rename_gene_dictionary(mdata, dictionary_file_path):
    """
    Map Ensembl gene IDs to gene symbols using a provided dictionary file.
    
    Rename genes in RNA modality using a dictionary file mapping Ensembl IDs to gene symbols.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing RNA modality with var_names to be renamed
    dictionary_file_path : str
        Path to TSV file with columns 'ensembl_id' and 'gene'
    """

    # Convert mapping result to list before assignment
    df = pd.read_csv(dictionary_file_path, sep='\t')
    ensemble_to_gene = dict(zip(df['ensembl_id'], df['gene']))
    new_names = [ensemble_to_gene.get(x, x) for x in mdata['rna'].var_names]
    mdata['rna'].var_names = (new_names)

