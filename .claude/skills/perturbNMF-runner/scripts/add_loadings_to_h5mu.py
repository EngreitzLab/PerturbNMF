#!/usr/bin/env python3
"""Add cNMF gene loadings to an existing h5mu, in place, without rewriting the file.

An h5mu assembled by ``build_cnmf_h5mu_from_usages.py`` before it learned
``--gene_spectra_score`` has an empty ``varm`` on the program modality, so Stage 2a
enrichment and every gene-level Stage 3 plot fail with ``KeyError: 'loadings'``.

Rebuilding is correct but expensive when ``--include_rna`` was used (the source gene
matrix can be tens of GB). The loadings are tiny (K x genes) and live in their own HDF5
groups, so this script appends them to the existing file instead:

    mod/<prog_key>/varm/loadings     (K x genes, float64)
    mod/<prog_key>/uns/var_names     (gene names, matching the spectra columns)

Nothing existing is deleted or overwritten unless ``--overwrite`` is passed, and the
program ids / gene count are validated against the file before anything is written.

Example
-------
python3 add_loadings_to_h5mu.py \
    --h5mu Result/Adam_run/Inference/adata/cNMF_200_2_0.h5mu \
    --gene_spectra_score Result/Adam_run/Inference/Inference.gene_spectra_score.k_200.dt_2_0.txt
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd

try:
    from anndata.experimental import read_elem, write_elem
except ImportError:  # anndata >= 0.11 moved them
    from anndata._io.specs import read_elem, write_elem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5mu', required=True, help='Existing .h5mu to patch in place')
    p.add_argument('--gene_spectra_score', required=True,
                   help='cNMF gene_spectra_score TSV (K x genes)')
    p.add_argument('--prog_key', default='cNMF', help='Program modality to patch')
    p.add_argument('--data_key', default='rna',
                   help='Gene modality to validate the spectra width against '
                        '(skipped if absent)')
    p.add_argument('--overwrite', action='store_true',
                   help="Replace varm['loadings'] / uns['var_names'] if already present")
    p.add_argument('--dry_run', action='store_true',
                   help='Validate and report, but write nothing')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Reading gene spectra score: {args.gene_spectra_score}', flush=True)
    loadings = pd.read_csv(args.gene_spectra_score, sep='\t', index_col=0)
    loadings.index = loadings.index.astype(str)
    print(f'  loadings: {loadings.shape[0]} programs x {loadings.shape[1]} genes',
          flush=True)

    mode = 'r' if args.dry_run else 'r+'
    with h5py.File(args.h5mu, mode) as f:
        mod = f'mod/{args.prog_key}'
        if mod not in f:
            raise ValueError(f"modality '{args.prog_key}' not found in {args.h5mu} "
                             f"(have: {list(f['mod'].keys())})")

        prog_ids = read_elem(f[f'{mod}/var']).index.astype(str)
        if set(prog_ids) != set(loadings.index):
            raise ValueError(
                f'program ids differ between the h5mu ({len(prog_ids)}) and the '
                f'spectra score ({loadings.shape[0]}): e.g. missing from spectra '
                f'{sorted(set(prog_ids) - set(loadings.index))[:5]}')
        # Row i of varm must be program var_names[i]; reindex rather than trust order.
        loadings = loadings.loc[prog_ids]

        data_var = f'mod/{args.data_key}/var'
        if data_var in f:
            n_genes = read_elem(f[data_var]).shape[0]
            if n_genes != loadings.shape[1]:
                raise ValueError(
                    f"the spectra score has {loadings.shape[1]} genes but modality "
                    f"'{args.data_key}' has {n_genes} -- Stage 2a asserts these match")
            print(f"  validated against '{args.data_key}': {n_genes} genes", flush=True)
        else:
            print(f"  note: no '{args.data_key}' modality, skipping the gene-count check",
                  flush=True)

        targets = [(f'{mod}/varm', 'loadings'), (f'{mod}/uns', 'var_names')]
        existing = [f'{g}/{k}' for g, k in targets if f'{g}/{k}' in f]
        if existing and not args.overwrite:
            raise ValueError(f'already present: {existing}. Pass --overwrite to replace.')

        if args.dry_run:
            print('Dry run: validation passed, nothing written.', flush=True)
            return 0

        for group, key in targets:
            if f'{group}/{key}' in f:
                print(f'  overwriting {group}/{key}', flush=True)
                del f[f'{group}/{key}']

        print(f'Writing {mod}/varm/loadings and {mod}/uns/var_names', flush=True)
        write_elem(f[f'{mod}/varm'], 'loadings',
                   np.asarray(loadings.values, dtype=np.float64))
        write_elem(f[f'{mod}/uns'], 'var_names',
                   loadings.columns.astype(str).to_numpy())

    print(f'Done. {args.h5mu} now carries loadings for {loadings.shape[0]} programs '
          f'x {loadings.shape[1]} genes.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
