#!/usr/bin/env python3
"""
Prepare guide data h5ad for CRT calibration.

Extracts perturbation/guide assignment info from an h5mu file and saves it
as a standalone h5ad with standardized key names (guide_assignment, guide_names,
guide_targets) compatible with CRT.py.

This is needed when h5mu files use alternative key names (e.g., target_assignment,
target_names) instead of the standard guide_* convention expected by CRT.py.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Prepare guide data h5ad for CRT calibration from h5mu"
    )
    parser.add_argument(
        '--h5mu_path', type=str, required=True,
        help='Path to input h5mu file (any K value works; use smallest for speed)'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Path to save output guide h5ad file'
    )
    parser.add_argument(
        '--modality', type=str, default='cNMF',
        help='Modality to extract guide info from (default: cNMF)'
    )
    parser.add_argument(
        '--assignment_key', type=str, default='target_assignment',
        help='Source key in obsm for guide assignment matrix (default: target_assignment)'
    )
    parser.add_argument(
        '--names_key', type=str, default='target_names',
        help='Source key in uns for guide/target names (default: target_names)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite output file if it exists'
    )

    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        print(f"Output file already exists: {args.output_path}")
        print("Use --force to overwrite. Skipping.")
        return

    import muon as mu
    import anndata as ad

    print(f"Reading h5mu: {args.h5mu_path}")
    mdata = mu.read(args.h5mu_path)
    mod = mdata[args.modality]

    print(f"Extracting guide info from '{args.modality}' modality...")
    print(f"  Source assignment key: obsm['{args.assignment_key}']")
    print(f"  Source names key: uns['{args.names_key}']")

    # Create minimal AnnData with guide info
    adata_guide = ad.AnnData(
        X=np.zeros((mod.n_obs, 0)),
        obs=pd.DataFrame(index=mod.obs_names),
    )

    # Copy all obs columns (needed for covariates in CRT)
    for col in mod.obs.columns:
        adata_guide.obs[col] = mod.obs[col].values

    # Map source keys to standard guide_* keys
    adata_guide.obsm['guide_assignment'] = np.array(mod.obsm[args.assignment_key])
    adata_guide.uns['guide_names'] = list(mod.uns[args.names_key])
    adata_guide.uns['guide_targets'] = np.array(mod.uns[args.names_key])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    adata_guide.write(args.output_path)

    print(f"Guide h5ad saved: {args.output_path}")
    print(f"  Cells: {adata_guide.n_obs}")
    print(f"  Targets: {len(adata_guide.uns['guide_names'])}")
    names = adata_guide.uns['guide_names']
    preview = names[:5] if len(names) > 5 else names
    print(f"  Target names (first 5): {preview}")


if __name__ == '__main__':
    main()
