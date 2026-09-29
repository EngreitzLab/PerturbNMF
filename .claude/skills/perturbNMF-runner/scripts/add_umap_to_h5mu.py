#!/usr/bin/env python3
"""Precompute PCA/UMAP into a cNMF h5mu so Stage 3b/3c plotting jobs skip it.

Both plotting entry points call ``ensure_umap(mdata, data_key, prog_key)`` right
after reading the h5mu (``cNMF_program_analysis.py``, ``cNMF_perturbed_gene_analysis.py``).
That helper computes PCA/neighbors/UMAP on the **full** cell set -- ``--subsample_frac``
only thins the scatter plots afterwards -- and it writes nothing back to disk. So on a
~1M-cell run every plotting job repays the same multi-hour UMAP, once for the program
PDF and again for the gene PDF.

This script does that work once and persists it. ``ensure_umap`` is imported and called
directly rather than reimplemented, so the stored ``X_umap``/``X_pca`` are by
construction identical to what the plotting scripts would have computed, under the same
top-variance gene selection (``--n_top_genes``, default 2000) and ``--n_comps``
(default 50). Both modalities get the arrays, which is exactly the state
``ensure_umap``'s first branch treats as a no-op.

Writing is opt-in: pass ``--output`` for a new file, or ``--in_place`` to replace the
input. ``--in_place`` writes a sibling ``.tmp`` first and then ``os.replace``s it, so an
interrupted job cannot leave a half-written h5mu -- but it does need free space for a
second copy of the file while it runs.

Example
-------
python3 add_umap_to_h5mu.py \
    --mdata_path Result/<run>/Inference/adata/cNMF_50_2_0.h5mu \
    --in_place --data_key rna --prog_key cNMF
"""

import argparse
import os
import sys
import time

import mudata as mu

# Same import convention as the plotting entry points, but resolved relative to this
# file rather than hardcoded, since this script sits under .claude/skills/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', 'src'))
sys.path.append(_PIPELINE_SRC)

from Stage3_Interpretation.A_Plotting.src import ensure_umap  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mdata_path', required=True, help='Input .h5mu')
    p.add_argument('--output', default=None,
                   help='Write the result here. Mutually exclusive with --in_place.')
    p.add_argument('--in_place', action='store_true',
                   help='Overwrite --mdata_path (via a .tmp file + atomic replace)')
    p.add_argument('--data_key', default='rna', help='Expression modality')
    p.add_argument('--prog_key', default='cNMF', help='Program modality')
    p.add_argument('--n_top_genes', type=int, default=2000,
                   help="Top-variance genes used as the UMAP basis. Must match the "
                        "plotting scripts' ensure_umap default (2000) or they will "
                        'recompute instead of reusing this.')
    p.add_argument('--n_comps', type=int, default=50, help='PCA components (default 50)')
    p.add_argument('--force', action='store_true',
                   help='Recompute even if X_umap is already present in both modalities')
    args = p.parse_args()
    if bool(args.output) == bool(args.in_place):
        p.error('pass exactly one of --output or --in_place')
    return args


def main():
    args = parse_args()
    t0 = time.time()

    print(f'Reading {args.mdata_path}', flush=True)
    mdata = mu.read_h5mu(args.mdata_path)
    print(f'  modalities: {list(mdata.mod)}', flush=True)

    for key in (args.data_key, args.prog_key):
        if key not in mdata.mod:
            raise KeyError(
                f"modality '{key}' not in {list(mdata.mod)}. The expression matrix is "
                'required -- a cNMF-only h5mu must be rebuilt with '
                '`build_cnmf_h5mu_from_usages.py --include_rna --tpm_rna` first.')

    present = ['X_umap' in mdata[k].obsm for k in (args.data_key, args.prog_key)]
    if all(present) and not args.force:
        print('X_umap already present in both modalities; nothing to do '
              '(pass --force to recompute).', flush=True)
        return 0
    if args.force:
        for k in (args.data_key, args.prog_key):
            for slot in ('X_umap', 'X_pca'):
                mdata[k].obsm.pop(slot, None)

    n_obs = mdata[args.data_key].n_obs
    print(f'Computing PCA({args.n_comps}) + neighbors + UMAP on {n_obs} cells from the '
          f'top {args.n_top_genes} variance genes', flush=True)
    ensure_umap(mdata, args.data_key, args.prog_key,
                n_top_genes=args.n_top_genes, n_comps=args.n_comps)
    print(f'  done in {time.time() - t0:.0f}s', flush=True)

    for k in (args.data_key, args.prog_key):
        if 'X_umap' not in mdata[k].obsm:
            raise RuntimeError(f"ensure_umap did not populate X_umap on '{k}'")
        print(f"  {k}.obsm['X_umap']: {mdata[k].obsm['X_umap'].shape}", flush=True)

    target = args.output or args.mdata_path
    tmp = f'{target}.tmp' if args.in_place else target
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    print(f'Writing {tmp}', flush=True)
    mdata.write(tmp)
    if args.in_place:
        print(f'Replacing {target}', flush=True)
        os.replace(tmp, target)

    print(f'Done in {time.time() - t0:.0f}s -> {target}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
