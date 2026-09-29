#!/usr/bin/env python3
"""Convert a raw IGVF mudata (gene + guide modalities) into a Stage 1 inference h5ad.

The IGVF deliverable keeps expression and guides in separate modalities, but the
inference pipelines expect a single AnnData carrying the guide data alongside the
counts::

    obsm['guide_assignment']   binary cells x guides matrix
    uns['guide_names']         guide ids   (columns of guide_assignment)
    uns['guide_targets']       target gene per guide
    obs[<condition_key>]       categorical used as --categorical_key

Important: IGVF guide modalities store **raw guide UMI counts in X** and the called
**binary assignment in layers['guide_assignment']**. This script uses the layer; using
X would treat ambient guide reads as perturbations.

Generalises the one-off ``Convert_file_adata.py`` written for IGVF_Gersbach_iHep.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='Raw IGVF .h5mu')
    p.add_argument('--output', required=True, help='Output .h5ad for --counts_fn')
    p.add_argument('--gene_mod', default='gene', help='Expression modality')
    p.add_argument('--guide_mod', default='guide', help='Guide modality')
    p.add_argument('--guide_layer', default='guide_assignment',
                   help='Layer with the binary assignment; "X" to use guide X instead')
    p.add_argument('--guide_id_col', default='guide_id', help='Guide id column in guide .var')
    p.add_argument('--guide_target_col', default='gene_name',
                   help='Target column in guide .var')
    p.add_argument('--guide_type_col', default='type',
                   help='Column flagging control guides (blank to skip relabelling)')
    p.add_argument('--control_type', nargs='*', default=['non-targeting'],
                   help='Values of the type column treated as controls')
    p.add_argument('--control_label', default='non-targeting',
                   help='Target label written for control guides')
    p.add_argument('--condition_key', default='condition',
                   help='Name of the constant obs column to add (--categorical_key)')
    p.add_argument('--condition_value', default='all', help='Value of that column')
    p.add_argument('--gene_symbol_col', default='symbol',
                   help='Column in gene .var with symbols; var_names are set from it '
                        'and the original index is kept in var["gene_ids"]')
    p.add_argument('--keep_ensembl_index', action='store_true',
                   help='Leave var_names as-is instead of switching to symbols')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Reading {args.input}', flush=True)
    mdata = mu.read(args.input)
    adata = mdata[args.gene_mod].copy()
    guide = mdata[args.guide_mod]
    print(f'  gene: {adata.shape[0]} cells x {adata.shape[1]} genes', flush=True)
    print(f'  guide: {guide.shape[1]} guides', flush=True)

    # --- guide assignment (binary layer, not raw UMI counts in X) ---
    if args.guide_layer != 'X' and args.guide_layer in guide.layers:
        assignment = guide.layers[args.guide_layer].copy()
        print(f"  using guide layer '{args.guide_layer}'", flush=True)
    else:
        assignment = guide.X.copy()
        print(f"  WARNING: layer '{args.guide_layer}' not found, using guide X -- "
              f'verify it holds assignments, not raw UMI counts', flush=True)

    gmax = assignment.data.max() if assignment.nnz else 0
    print(f'  assignment: {assignment.nnz} nonzeros, max {gmax}, '
          f'{assignment.nnz / assignment.shape[0]:.2f} guides/cell', flush=True)
    if gmax > 1:
        print('  WARNING: assignment matrix is not binary (max > 1)', flush=True)

    adata.obsm['guide_assignment'] = assignment

    # --- guide names / targets ---
    if args.guide_id_col in guide.var.columns:
        guide_names = guide.var[args.guide_id_col].astype(str).to_numpy()
    else:
        guide_names = guide.var_names.astype(str).to_numpy()
    guide_targets = guide.var[args.guide_target_col].astype(object).to_numpy()

    if args.guide_type_col and args.guide_type_col in guide.var.columns:
        is_control = guide.var[args.guide_type_col].astype(str).isin(
            args.control_type).to_numpy()
        guide_targets[is_control] = args.control_label
        print(f'  relabelled {int(is_control.sum())} control guides as '
              f'"{args.control_label}"', flush=True)

    unlabelled = pd.isna(guide_targets)
    if unlabelled.any():
        print(f'  WARNING: {int(unlabelled.sum())} guides have no target -> "unassigned"',
              flush=True)
        guide_targets[unlabelled] = 'unassigned'

    adata.uns['guide_names'] = list(np.asarray(guide_names, dtype=str))
    adata.uns['guide_targets'] = list(np.asarray(guide_targets, dtype=str))

    # --- condition column used as --categorical_key ---
    adata.obs[args.condition_key] = pd.Categorical(
        [args.condition_value] * adata.shape[0])

    # --- gene names ---
    if not args.keep_ensembl_index and args.gene_symbol_col in adata.var.columns:
        adata.var['gene_ids'] = adata.var_names
        adata.var_names = adata.var[args.gene_symbol_col].astype(str)
        adata.var_names_make_unique()
        print(f'  var_names set from var["{args.gene_symbol_col}"]', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f'Writing {args.output}', flush=True)
    adata.write(args.output)
    print(f'Done. {adata.shape[0]} cells x {adata.shape[1]} genes, '
          f'{len(set(adata.uns["guide_targets"]))} unique targets.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
