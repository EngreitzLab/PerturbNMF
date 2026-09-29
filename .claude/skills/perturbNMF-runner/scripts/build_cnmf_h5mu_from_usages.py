#!/usr/bin/env python3
"""Assemble a PerturbNMF-style cNMF h5mu from a consensus usages TSV + a raw IGVF mudata.

Some cNMF runs are only available as their published artifacts: a consensus usages
matrix (cells x K) next to the raw IGVF input mudata (``gene`` + ``guide`` modalities).
Stage 2b (CRT, U-test) and Stage 3 plotting all expect
``<out_dir>/<run_name>/Inference/adata/cNMF_<K>_<thresh>.h5mu`` with a ``cNMF`` modality
carrying the usages in ``X`` plus the guide metadata that ``reformat_data_for_CRT``
reads (``obsm['guide_assignment']``, ``uns['guide_names']``, ``uns['guide_targets']``).

This script builds exactly that. Pass ``--gene_spectra_score`` to also carry the gene
loadings over as ``varm['loadings']`` + ``uns['var_names']`` -- Stage 2a enrichment and
every gene-level Stage 3 plot read those, and without them the program-analysis plot
dies with ``KeyError: 'loadings'``.

The raw gene expression matrix is deliberately NOT
copied -- CRT never touches it, and it is usually the bulk of the input file. Pass
``--include_rna`` to carry it over anyway (needed later for Stage 2a evaluation and
Stage 3b/3c plots), and add ``--tpm_rna`` when the source matrix holds raw integer
counts -- Stage 3b/3c call ``check_normalized()`` and raise on those. ``--tpm_rna``
uses cNMF's own TPM definition (``compute_tpm``: ``normalize_total(target_sum=1e6)``,
no ``log1p``), so the carried-over matrix is on the same scale as inference used.

Example
-------
python3 build_cnmf_h5mu_from_usages.py \
    --usages     Result/Adam_run/Inference/Inference.usages.k_200.dt_2_0.consensus.txt \
    --gene_spectra_score Result/Adam_run/Inference/Inference.gene_spectra_score.k_200.dt_2_0.txt \
    --raw_mudata Result/Adam_run/Data/inference_mudata.h5mu \
    --output     Result/Adam_run/Inference/adata/cNMF_200_2_0.h5mu \
    --include_rna --tpm_rna \
    --control_type non-targeting --condition_value all
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu

try:
    from anndata.experimental import read_elem
except ImportError:  # anndata >= 0.11 moved it
    from anndata._io.specs import read_elem


def _xmax(X):
    """Max of a dense or sparse matrix, 0 for an all-zero sparse one."""
    data = getattr(X, 'data', X)
    return float(np.asarray(data).max()) if np.size(data) else 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--usages', required=True,
                   help='Consensus usages TSV (cells x K), cell barcodes in the index')
    p.add_argument('--gene_spectra_score',
                   help="cNMF gene_spectra_score TSV (K x genes), the z-scored gene "
                        "loadings -> varm['loadings'] + uns['var_names']. Stage 2a "
                        "enrichment and every Stage 3 gene-level plot read these; "
                        "without them plot_top_gene_per_program raises "
                        "KeyError: 'loadings'.")
    p.add_argument('--raw_mudata', required=True,
                   help='Raw IGVF .h5mu with gene + guide modalities')
    p.add_argument('--output', required=True,
                   help='Output .h5mu path (e.g. .../adata/cNMF_200_2_0.h5mu)')
    p.add_argument('--gene_mod', default='gene', help='Modality holding expression + obs')
    p.add_argument('--guide_mod', default='guide', help='Modality holding the guide matrix')
    p.add_argument('--guide_layer', default='guide_assignment',
                   help='Layer in the guide modality holding the BINARY assignment matrix. '
                        'IGVF mudatas keep raw guide UMI counts in X and the called '
                        'assignment in this layer -- using X would treat ambient guide '
                        'reads as perturbations. Pass "X" to use X anyway.')
    p.add_argument('--prog_key', default='cNMF', help='Name of the program modality to write')
    p.add_argument('--guide_id_col', default='guide_id',
                   help="Column in guide .var with the guide id -> uns['guide_names']")
    p.add_argument('--guide_target_col', default='gene_name',
                   help="Column in guide .var with the target -> uns['guide_targets']")
    p.add_argument('--guide_type_col', default='type',
                   help='Column in guide .var flagging control guides')
    p.add_argument('--control_type', nargs='*', default=['non-targeting'],
                   help='Value(s) of the type column whose targets get relabelled')
    p.add_argument('--control_label', default='non-targeting',
                   help='Target label written for control guides')
    p.add_argument('--condition_key', default='condition',
                   help='Name of the constant condition column added to obs')
    p.add_argument('--condition_value', default='all',
                   help='Value of the constant condition column')
    p.add_argument('--include_rna', action='store_true',
                   help='Also carry the gene modality over as an "rna" modality')
    p.add_argument('--tpm_rna', action='store_true',
                   help='TPM-normalize the carried-over rna matrix, using the same '
                        'definition cNMF itself uses in compute_tpm() -- '
                        'normalize_total(target_sum=1e6), no log1p. Raw IGVF gene '
                        'matrices hold integer counts, which Stage 3b/3c reject via '
                        'check_normalized().')
    p.add_argument('--data_key', default='rna', help='Modality name used with --include_rna')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Reading usages: {args.usages}', flush=True)
    usages = pd.read_csv(args.usages, sep='\t', index_col=0)
    usages.index = usages.index.astype(str)
    print(f'  usages: {usages.shape[0]} cells x {usages.shape[1]} programs', flush=True)

    loadings = None
    if args.gene_spectra_score:
        print(f'Reading gene spectra score: {args.gene_spectra_score}', flush=True)
        loadings = pd.read_csv(args.gene_spectra_score, sep='\t', index_col=0)
        loadings.index = loadings.index.astype(str)
        print(f'  loadings: {loadings.shape[0]} programs x {loadings.shape[1]} genes',
              flush=True)
        # The spectra file and the usages file are both keyed by program id, but there
        # is no guarantee the two are written in the same order -- reindex rather than
        # trust it, so varm['loadings'] row i really is program var_names[i].
        prog_ids = usages.columns.astype(str)
        if set(prog_ids) != set(loadings.index):
            raise ValueError(
                f'program ids differ between usages ({len(prog_ids)}) and the spectra '
                f'score ({loadings.shape[0]}): e.g. missing from spectra '
                f'{sorted(set(prog_ids) - set(loadings.index))[:5]}')
        loadings = loadings.loc[prog_ids]

    print(f'Reading raw mudata (metadata only): {args.raw_mudata}', flush=True)
    with h5py.File(args.raw_mudata, 'r') as f:
        gene_obs = read_elem(f[f'mod/{args.gene_mod}/obs'])
        guide_var = read_elem(f[f'mod/{args.guide_mod}/var'])
        # Prefer the binary assignment layer over X (raw guide UMI counts).
        layer_path = f'mod/{args.guide_mod}/layers/{args.guide_layer}'
        if args.guide_layer != 'X' and layer_path in f:
            print(f"  using guide layer '{args.guide_layer}'", flush=True)
            guide_X = read_elem(f[layer_path])
        else:
            print(f"  WARNING: layer '{args.guide_layer}' not found, falling back to "
                  f'guide X -- verify it holds assignments, not raw UMI counts', flush=True)
            guide_X = read_elem(f[f'mod/{args.guide_mod}/X'])
        rna_X = rna_var = None
        if args.include_rna:
            rna_X = read_elem(f[f'mod/{args.gene_mod}/X'])
            rna_var = read_elem(f[f'mod/{args.gene_mod}/var'])
    gene_obs.index = gene_obs.index.astype(str)
    print(f'  obs: {gene_obs.shape[0]} cells x {gene_obs.shape[1]} columns', flush=True)
    print(f'  guides: {guide_X.shape[1]}', flush=True)

    # --- align cells: cNMF usually drops a few cells during consensus ---
    missing = usages.index.difference(gene_obs.index)
    if len(missing):
        raise ValueError(
            f'{len(missing)} cells in the usages file are absent from the mudata '
            f'(e.g. {list(missing[:5])}). The usages and mudata do not match.')
    dropped = len(gene_obs.index) - len(usages.index)
    print(f'  aligning to the usages index ({dropped} mudata cells dropped by cNMF)',
          flush=True)

    positions = gene_obs.index.get_indexer(usages.index)
    obs = gene_obs.iloc[positions].copy()
    guide_assignment = guide_X[positions, :]

    obs[args.condition_key] = pd.Categorical([args.condition_value] * obs.shape[0])

    # A guide-assignment matrix should be binary; raw UMI counts are a common mix-up.
    gmax = guide_assignment.data.max() if guide_assignment.nnz else 0
    print(f'  guide assignment: {guide_assignment.nnz} nonzeros, max value {gmax}, '
          f'{guide_assignment.nnz / guide_assignment.shape[0]:.2f} guides/cell', flush=True)
    if gmax > 1:
        print('  WARNING: assignment matrix is not binary (max > 1) -- this looks like '
              'raw guide UMI counts, not called assignments', flush=True)

    # --- guide metadata ---
    if args.guide_id_col in guide_var.columns:
        guide_names = guide_var[args.guide_id_col].astype(str).to_numpy()
    else:
        guide_names = guide_var.index.astype(str).to_numpy()
    guide_targets = guide_var[args.guide_target_col].astype(object).to_numpy()

    is_control = guide_var[args.guide_type_col].astype(str).isin(args.control_type).to_numpy()
    guide_targets[is_control] = args.control_label
    print(f'  relabelled {int(is_control.sum())} control guides as '
          f'"{args.control_label}"', flush=True)

    # Anything still missing a target (NaN) would silently become the string "nan"
    # downstream, so surface it here instead.
    unlabelled = pd.isna(guide_targets)
    if unlabelled.any():
        print(f'  WARNING: {int(unlabelled.sum())} guides have no target and will be '
              f'labelled "unassigned"', flush=True)
        guide_targets[unlabelled] = 'unassigned'
    guide_targets = guide_targets.astype(str)

    if len(guide_names) != guide_assignment.shape[1]:
        raise ValueError(f'guide_names ({len(guide_names)}) does not match the guide '
                         f'matrix width ({guide_assignment.shape[1]})')

    # --- program modality ---
    prog = ad.AnnData(
        X=np.asarray(usages.values, dtype=np.float64),
        obs=obs,
        var=pd.DataFrame(index=usages.columns.astype(str)),
    )
    prog.obsm['guide_assignment'] = guide_assignment.tocsr()
    prog.uns['guide_names'] = guide_names
    prog.uns['guide_targets'] = guide_targets

    if loadings is not None:
        # Stage 2a asserts loadings.shape[1] == mdata[data_key].var.shape[0], so the
        # spectra columns have to line up with the gene modality one-for-one.
        if rna_var is not None and loadings.shape[1] != rna_var.shape[0]:
            raise ValueError(
                f'the spectra score has {loadings.shape[1]} genes but the '
                f'{args.gene_mod} modality has {rna_var.shape[0]} -- Stage 2a/3 '
                f'assume they are the same genes in the same order')
        prog.varm['loadings'] = np.asarray(loadings.values, dtype=np.float64)
        prog.uns['var_names'] = loadings.columns.astype(str).to_numpy()

    mods = {args.prog_key: prog}
    if args.include_rna:
        rna = ad.AnnData(X=rna_X[positions, :], obs=obs.copy(), var=rna_var)
        if args.tpm_rna:
            import scanpy as sc
            # Same normalization cNMF applies in compute_tpm() (cnmf/cnmf.py), done
            # in place rather than via that helper so a ~19GB matrix is not copied.
            print(f'  TPM-normalizing rna (max before: {_xmax(rna.X):g})', flush=True)
            sc.pp.normalize_total(rna, target_sum=1e6)
            print(f'  rna TPM-normalized (max after: {_xmax(rna.X):g})', flush=True)
        rna.obsm['guide_assignment'] = guide_assignment.tocsr()
        rna.uns['guide_names'] = guide_names
        rna.uns['guide_targets'] = guide_targets
        mods[args.data_key] = rna

    mdata = mu.MuData(mods)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f'Writing {args.output}', flush=True)
    mdata.write(args.output)

    n_ctrl = int((guide_targets == args.control_label).sum())
    print(f'Done. {args.prog_key}: {prog.shape[0]} cells x {prog.shape[1]} programs; '
          f'{len(np.unique(guide_targets))} unique targets; '
          f'{n_ctrl} control guides labelled "{args.control_label}".', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
