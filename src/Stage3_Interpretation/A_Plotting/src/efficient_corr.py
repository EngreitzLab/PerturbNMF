"""Memory-efficient Pearson correlations for the Stage 3 plotting reports.

Only the correlations that are too big for plain pandas live here:

- ``RowCorr``: large N (genes, perturbed regulators). Keeps only the
  row-z-scored table Z (N x K) and computes one correlation row on demand as
  ``Z @ Z[i] / K``, so the N x N matrix (3.6 GB at N=30k) is never held in
  memory while plotting. ``full()`` builds the full matrix once to save it.
- ``column_pearson_streaming``: K x K correlation between the columns of a
  cells x K usage matrix, streamed over cells.

Callers save the matrices with plain ``np.savez`` (``corr`` + a names array
of plain strings, so ``np.load`` needs no ``allow_pickle``).
"""

import numpy as np
import pandas as pd


class RowCorr:
    """Pearson correlation between the rows of an (N, K) table, one row at a time.

    Keeps only Z, the table with each row z-scored across its K columns
    (16 MB at N=20k, K=200), and computes a row of the N x N correlation as
    ``Z @ Z[i] / K`` when asked. Supports the DataFrame calls the plots use:
    ``corr.loc[name]``, ``corr[name]``, ``name in corr``, ``.index`` /
    ``.columns``, ``.copy()``, ``.shape``.

    Parameters
    ----------
    X : array-like, shape (N, K)
        Rows are the items to correlate (genes, regulators).
    names : sequence of str
        Row names, length N.
    row_label, col_label : str
        Used only in the missing-value error message.
    """

    def __init__(self, X, names, row_label="rows", col_label="columns"):
        self.Z = np.ascontiguousarray(_zscore_rows(X, names, row_label, col_label),
                                      dtype=np.float32)
        self.index = self.columns = pd.Index(np.asarray(names).astype(str), dtype=object)
        self._pos = {g: i for i, g in enumerate(self.index)}

    @property
    def shape(self):
        return (len(self.index), len(self.index))

    def __getitem__(self, name):
        i = self._pos[name]
        r = self.Z @ self.Z[i] / self.Z.shape[1]
        return pd.Series(r, index=self.index, name=name)

    @property
    def loc(self):
        return self  # corr.loc[name] == corr[name]

    def __contains__(self, name):
        return name in self._pos

    def copy(self):
        return self

    def full(self):
        """The full N x N float32 matrix, e.g. to save with ``np.savez``.

        Allocates N**2 * 4 bytes (1.6 GB at N=20k, 3.6 GB at N=30k). Unlike
        the plotting rows, NaN is not filled and the diagonal is kept.
        """
        R = self.Z @ self.Z.T
        R /= self.Z.shape[1]
        return R


def _zscore_rows(X, names, row_label="rows", col_label="columns"):
    """Z-score each row of ``X`` (ddof=0) so that row Pearson = mean(Z[i] * Z[j]).

    Raises ValueError if any row has a missing value (the shortcut needs a
    complete table). Rows with a constant profile become NaN, so their
    correlations are NaN, as in ``pd.DataFrame.corr()``.
    """
    X = np.asarray(X, dtype=np.float64)
    missing = np.isnan(X).any(axis=1)
    if missing.any():
        bad = pd.Index(names)[missing].tolist()
        raise ValueError(
            f"{len(bad)} {row_label} have missing values for some {col_label} "
            f"(e.g. {bad[:5]}); a complete {row_label} x {col_label} table is required."
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    Z[np.ptp(X, axis=1) == 0] = np.nan  # exact test; round-off can leave std tiny but nonzero
    return Z


def column_pearson_streaming(X, chunk_rows=100_000):
    """K x K Pearson correlation between the columns of an (N, K) matrix.

    Two passes over ``chunk_rows``-row chunks (column means, then the centered
    ``C.T @ C`` accumulated in float64), so memory is one chunk plus K x K
    instead of the full N x K copies made by ``pd.DataFrame(X).corr()``.
    ``X`` may be dense, scipy-sparse or h5py-backed. Constant columns -> NaN.
    """
    N, K = X.shape

    def chunks():
        for s in range(0, N, chunk_rows):
            C = X[s:s + chunk_rows]
            C = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
            yield C.astype(np.float64, copy=False)

    total = np.zeros(K)
    lo, hi = np.full(K, np.inf), np.full(K, -np.inf)
    for C in chunks():
        total += C.sum(axis=0)
        lo, hi = np.minimum(lo, C.min(axis=0)), np.maximum(hi, C.max(axis=0))
    mean = total / N
    S = np.zeros((K, K))
    for C in chunks():
        C = C - mean
        S += C.T @ C
    sd = np.sqrt(np.diag(S))
    sd[hi == lo] = np.nan  # exact constant test; round-off can leave sd tiny but nonzero
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / np.outer(sd, sd)
