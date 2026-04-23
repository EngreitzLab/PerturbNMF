"""
Subsample the full CC-Perturb-seq h5ad to create a mini test dataset.

Takes 500 cells per condition (batch), subsets guide_assignment to non-empty
guides, and saves to tests/data/mini_ccperturb.h5ad.

Usage:
    conda activate sk-cNMF
    python tests/create_mini_dataset.py
"""

import numpy as np
import anndata as ad
import scipy.sparse as sp
from pathlib import Path

FULL_DATA = "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Data/raw_updated_withguide_030526.h5ad"
OUTPUT = Path(__file__).parent / "data" / "mini_ccperturb.h5ad"
CELLS_PER_CONDITION = 500
SEED = 42


def main():
    print(f"Loading {FULL_DATA} ...")
    adata = ad.read_h5ad(FULL_DATA)
    print(f"Full shape: {adata.shape}")

    # Sample 500 cells per batch condition
    rng = np.random.default_rng(SEED)
    idx = []
    for batch in sorted(adata.obs["batch"].unique()):
        batch_idx = np.where(adata.obs["batch"] == batch)[0]
        n = min(CELLS_PER_CONDITION, len(batch_idx))
        chosen = rng.choice(batch_idx, size=n, replace=False)
        idx.extend(chosen)
        print(f"  {batch}: sampled {n} cells")

    idx = sorted(idx)
    mini = adata[idx].copy()
    print(f"Subsampled shape: {mini.shape}")

    # Subset guide_assignment to non-empty guides
    ga = mini.obsm["guide_assignment"]
    if sp.issparse(ga):
        ga = ga.toarray()
    guide_mask = ga.sum(axis=0) > 0
    n_before = ga.shape[1]
    ga_sub = ga[:, guide_mask]
    mini.obsm["guide_assignment"] = ga_sub

    # Update guide_names and guide_targets to match
    guide_names = np.array(mini.uns["guide_names"])
    guide_targets = np.array(mini.uns["guide_targets"])
    mini.uns["guide_names"] = guide_names[guide_mask]
    mini.uns["guide_targets"] = guide_targets[guide_mask]

    n_after = guide_mask.sum()
    print(f"Guides: {n_before} -> {n_after} (dropped {n_before - n_after} empty)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    mini.write(str(OUTPUT))
    print(f"Saved to {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
