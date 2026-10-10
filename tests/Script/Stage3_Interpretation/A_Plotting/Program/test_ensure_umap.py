"""Tests for ensure_umap reusing embeddings stored under non-scanpy obsm keys.

Seurat conversions store embeddings as obsm['UMAP'] / obsm['PCA'] (often as
DataFrames); ensure_umap should reuse them instead of computing a new UMAP.

Usage (from the repo root):
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/Program/test_ensure_umap.py -v
"""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from Stage3_Interpretation.A_Plotting.src.utilities import ensure_umap

N_CELLS = 60
N_GENES = 40
N_PROGRAMS = 3
SEED = 0


def _mdata():
    """MuData with 'rna' and 'cNMF' modalities and no embeddings."""
    rng = np.random.default_rng(SEED)
    cells = [f"cell{i}" for i in range(N_CELLS)]
    rna = ad.AnnData(
        X=rng.poisson(3.0, size=(N_CELLS, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(N_GENES)]),
    )
    prog = ad.AnnData(
        X=rng.random((N_CELLS, N_PROGRAMS)).astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=[str(p) for p in range(N_PROGRAMS)]),
    )
    return md.MuData({"rna": rna, "cNMF": prog})


def _seurat_embedding(mdata, n_cols, prefix):
    rng = np.random.default_rng(SEED + n_cols)
    return pd.DataFrame(
        rng.normal(size=(N_CELLS, n_cols)),
        index=mdata["rna"].obs_names,
        columns=[f"{prefix}_{i + 1}" for i in range(n_cols)],
    )


def test_adopts_seurat_keys_and_mirrors_to_programs(capsys):
    mdata = _mdata()
    umap = _seurat_embedding(mdata, 3, "UMAP")  # some converters keep 3 dims
    pca = _seurat_embedding(mdata, 10, "PC")
    mdata["rna"].obsm["UMAP"] = umap
    mdata["rna"].obsm["PCA"] = pca

    ensure_umap(mdata, "rna", "cNMF")

    for key in ["rna", "cNMF"]:
        x_umap = mdata[key].obsm["X_umap"]
        assert isinstance(x_umap, np.ndarray)
        np.testing.assert_array_equal(x_umap, umap.to_numpy()[:, :2])
        np.testing.assert_array_equal(mdata[key].obsm["X_pca"], pca.to_numpy())
    assert "UMAP" in mdata["rna"].obsm and "PCA" in mdata["rna"].obsm
    assert "using obsm['UMAP'] as 'X_umap'" in capsys.readouterr().out


def test_existing_x_umap_is_not_overwritten():
    mdata = _mdata()
    existing = np.zeros((N_CELLS, 2))
    for key in ["rna", "cNMF"]:
        mdata[key].obsm["X_umap"] = existing
    mdata["rna"].obsm["UMAP"] = _seurat_embedding(mdata, 2, "UMAP")

    ensure_umap(mdata, "rna", "cNMF")

    for key in ["rna", "cNMF"]:
        np.testing.assert_array_equal(mdata[key].obsm["X_umap"], existing)


def test_computes_umap_when_no_embedding_exists():
    mdata = _mdata()

    ensure_umap(mdata, "rna", "cNMF", n_top_genes=N_GENES, n_comps=5)

    for key in ["rna", "cNMF"]:
        assert mdata[key].obsm["X_umap"].shape == (N_CELLS, 2)
        assert mdata[key].obsm["X_pca"].shape == (N_CELLS, 5)
