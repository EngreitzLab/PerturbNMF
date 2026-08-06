"""Rank-K factored stand-ins for gene-by-gene correlation matrices.

These shims avoid materializing the O(G^2) or O(T^2) dense correlation
matrix used by the perturbed-gene plotting helpers. Both Pearson correlation
matrices produced upstream are mathematically rank-K (K = number of cNMF
programs, typically ~50), so storing the K-row factor and reconstructing
one row at a time costs O(K*G) memory and O(K*G) compute per row instead
of O(G^2) for both.

Callers consume only ``.loc[gene]``, ``.columns`` / ``.index`` membership,
and ``.copy()``; those are the only DataFrame-API surfaces implemented here.
"""

import numpy as np
import pandas as pd


class LazyGeneCorr:
    """Rank-K factored stand-in for the G x G gene-loading correlation matrix.

    Stores the K x G centered + unit-norm factor ``M``. Any row of the full
    G x G Pearson correlation is reconstructed on demand as ``M[:, i].T @ M``
    (length G). Exposes the subset of the pd.DataFrame API used by callers:
    ``.loc[gene]``, ``corr[gene]``, and ``.columns`` membership testing.
    """

    def __init__(self, M, gene_names):
        self.M = np.ascontiguousarray(M, dtype=np.float32)
        self.gene_names = pd.Index(np.asarray(gene_names, dtype=object))
        self._idx = {g: i for i, g in enumerate(self.gene_names)}

    @property
    def columns(self):
        return self.gene_names

    @property
    def index(self):
        return self.gene_names

    @property
    def shape(self):
        return (len(self.gene_names), len(self.gene_names))

    def _row(self, gene):
        i = self._idx[gene]
        r = (self.M[:, i:i + 1].T @ self.M).ravel().astype(np.float32, copy=False)
        np.nan_to_num(r, copy=False)
        return pd.Series(r, index=self.gene_names, name=gene)

    def __contains__(self, gene):
        return gene in self._idx

    def __getitem__(self, gene):
        return self._row(gene)

    class _Locator:
        def __init__(self, parent):
            self._p = parent

        def __getitem__(self, gene):
            return self._p._row(gene)

    @property
    def loc(self):
        return LazyGeneCorr._Locator(self)

    def copy(self):
        return self

    def to_dense(self):
        """Materialize the full G x G correlation matrix as a DataFrame.

        Warning: allocates ``G**2 * 4`` bytes (e.g. 3.6 GB at G=30k,
        360 GB at G=300k). Use only when an external tool genuinely needs
        the explicit matrix.
        """
        corr = (self.M.T @ self.M).astype(np.float32, copy=False)
        np.nan_to_num(corr, copy=False)
        return pd.DataFrame(corr, index=self.gene_names, columns=self.gene_names)

    def save_npz(self, path):
        np.savez_compressed(
            path,
            M=self.M,
            gene_names=np.asarray(self.gene_names, dtype=object),
        )

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=True) as f:
            return cls(M=f["M"], gene_names=f["gene_names"])

    @classmethod
    def from_loadings(cls, X, gene_names):
        """Build a LazyGeneCorr from a (K, G) gene-loading matrix.

        Centers each gene's loading vector across programs and normalizes
        it to unit norm so that ``M[:, i].T @ M[:, j]`` is the Pearson
        correlation of the two gene vectors. Genes with zero variance
        (constant loadings) yield NaN correlations that ``_row`` then
        replaces with 0 to match the legacy ``.fillna(0)`` behavior.
        """
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0, keepdims=True)
        M = X - mean
        norm = np.linalg.norm(M, axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            M = M / norm
        M = np.where(np.isfinite(M), M, 0.0)
        return cls(M=M.astype(np.float32), gene_names=gene_names)


class LazyPerturbCorr:
    """Rank-K factored stand-in for the per-sample T x T perturbation correlation.

    Stores the (T, K) pivoted log2FC matrix plus a NaN-aware mask. Any row of
    the T x T pairwise-complete Pearson correlation is reconstructed on demand
    using mask-aware matmuls equivalent to ``pivot_df.T.corr().loc[gene]``.
    Self-correlation is set to NaN to match ``np.fill_diagonal(corr, NaN)``.
    """

    def __init__(self, M, mask, target_names):
        M0 = np.where(mask, M, 0.0).astype(np.float32)
        self.M0 = np.ascontiguousarray(M0)
        self.mask = np.ascontiguousarray(mask.astype(np.float32))
        self.target_names = pd.Index(np.asarray(target_names, dtype=object))
        self._idx = {g: i for i, g in enumerate(self.target_names)}

    @property
    def index(self):
        return self.target_names

    @property
    def columns(self):
        return self.target_names

    @property
    def shape(self):
        return (len(self.target_names), len(self.target_names))

    def _row(self, gene):
        i = self._idx[gene]
        v0 = self.M0[i]                  # (K,)
        mv = self.mask[i]                # (K,)
        n = self.mask @ mv               # (T,) pairwise-complete counts
        sx = self.M0 @ mv                # (T,)
        sy = self.mask @ v0              # (T,)
        sxy = self.M0 @ v0               # (T,)
        sxx = (self.M0 * self.M0) @ mv   # (T,)
        syy = self.mask @ (v0 * v0)      # (T,)
        n_safe = np.where(n > 0, n, 1.0)
        num = sxy - sx * sy / n_safe
        denx = np.clip(sxx - sx * sx / n_safe, 0.0, None)
        deny = np.clip(syy - sy * sy / n_safe, 0.0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / np.sqrt(denx * deny)
        r[n <= 1] = np.nan
        r[i] = np.nan  # self-correlation
        return pd.Series(r.astype(np.float32, copy=False), index=self.target_names, name=gene)

    def __contains__(self, gene):
        return gene in self._idx

    def __getitem__(self, gene):
        return self._row(gene)

    class _Locator:
        def __init__(self, parent):
            self._p = parent

        def __getitem__(self, gene):
            return self._p._row(gene)

    @property
    def loc(self):
        return LazyPerturbCorr._Locator(self)

    def copy(self):
        return self

    def save_npz(self, path):
        np.savez_compressed(
            path,
            M0=self.M0,
            mask=self.mask.astype(np.bool_),
            target_names=np.asarray(self.target_names, dtype=object),
        )

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=True) as f:
            M0 = f["M0"]
            mask = f["mask"].astype(bool)
            target_names = f["target_names"]
        return cls(M=M0, mask=mask, target_names=target_names)

    @classmethod
    def from_pivot(cls, pivot_df):
        """Build a LazyPerturbCorr from a (T, K) pivot DataFrame.

        Preserves the pairwise-complete NaN semantics of
        ``pivot_df.T.corr()``: cells missing in the pivot are masked out
        and excluded from each pairwise Pearson computation.
        """
        target_names = pivot_df.index
        M = pivot_df.to_numpy(dtype=np.float64, copy=False)
        mask = ~np.isnan(M)
        return cls(M=M, mask=mask, target_names=target_names)
