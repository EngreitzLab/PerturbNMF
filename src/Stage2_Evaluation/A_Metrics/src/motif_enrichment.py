"""Program TF-motif enrichment.

Method (the t-test method of Schnitzler et al., Nature 2024, which reports program motif enrichment
in its Supplementary Table 24):

1. FIMO hits per element type (promoter: sequence_name is the gene; enhancer: ABC element with a
   target gene). Hits kept at ``p-value < threshold`` (paper: 1e-4 promoter, 1e-6 enhancer).
2. Motif ids collapse to a TF name (text before the first ``_``: ``ZN329_HUMAN.H11MO.0.C`` -> ``ZN329``).
   Motif ids without ``_`` that carry a ``motif_alt_id`` (JASPAR: ``MA0139.1`` / ``CTCF``) use the alt id.
3. Gene x TF hit counts; enhancer hits are summed over all ABC enhancers of the target gene
   (ABC elements of class ``promoter`` are dropped).
4. Universe = expressed genes (program genes) with >= 1 hit for any TF of that element type.
   Program genes = top ``n_top`` genes by gene_spectra_score; those with 0 hits are dropped.
   Background = universe minus the program genes.
5. Per program x TF: Welch two-sided t-test of program counts vs background counts (R ``t.test``);
   enrichment = mean(program) / mean(background); BH FDR across all program x TF of the element type.
6. Significant = FDR < 0.05 and enrichment > 1.

Alternative statistic (``test_motif_enrichment_correlation``, the loading-correlation variant
from the older ``enrichment_motif.py``): per program x TF, Pearson or Spearman correlation of
the per-gene motif count with the full program loading vector across the universe genes (default:
all expressed genes, genes without hits count 0); BH across all program x TF of the element type.
Same long table; ``enrichment`` holds the correlation coefficient, so significant = FDR < 0.05 and
correlation > 0 (:data:`SIGNIFICANCE_MIN_ENRICHMENT`).

Fi-NeMo hit tables (``motif_hit_calling.call_hits_from_finemo``) have NA p-values: read them with
``read_fimo_hits(path, pvalue_threshold=None)``.
"""

from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

ENRICHMENT_METHODS = ("ttest", "correlation")
# enrichment must exceed this for a row to be significant: mean ratio > 1 (t-test), correlation > 0
SIGNIFICANCE_MIN_ENRICHMENT = {"ttest": 1.0, "correlation": 0.0}

RESULT_COLUMNS = [
    "program", "element_type", "tf", "pvalue", "fdr", "enrichment",
    "n_program_genes_tested", "n_background_genes",
    "mean_count_program", "mean_count_background",
]


# ---------------------------------------------------------------------------
# Motif hits -> gene x TF counts
# ---------------------------------------------------------------------------

def collapse_motif_to_tf(motif_ids: Iterable[str]) -> pd.Series:
    """TF name = text before the first '_' of a HOCOMOCO motif id (``AP1_HUMAN.H11MO.0.A`` -> ``AP1``)."""
    return pd.Series(motif_ids).astype(str).str.split("_", n=1).str[0]


def parse_abc_sequence_name(sequence_names: pd.Series) -> pd.DataFrame:
    """Split ABC FASTA names ``region|class|element|TargetGene`` into element_name, element_class, gene.

    ``class|element`` is the ABC ``name`` column (e.g. ``genic|chr1:1-500``); bedtools ``-name`` may
    append ``::chr:start-end`` to the last field, which is stripped.
    """
    parts = sequence_names.str.split("|", expand=True)
    if parts.shape[1] != 4:
        raise ValueError(f"expected 4 '|'-separated fields in ABC sequence names, got {parts.shape[1]}")
    return pd.DataFrame({
        "element_name": parts[0].to_numpy(),
        "element_class": parts[1].to_numpy(),
        "gene": parts[3].str.split("::", n=1).str[0].to_numpy(),
    })


def read_fimo_hits(
    path: str,
    pvalue_threshold: Optional[float],
    sequence_name_parser: Optional[Callable[[pd.Series], pd.DataFrame]] = None,
    chunksize: int = 5_000_000,
    use_motif_alt_id: bool = True,
    collapse_motif_ids: bool = True,
) -> pd.DataFrame:
    """Read a FIMO ``fimo.tsv`` and keep hits with ``p-value < pvalue_threshold`` (strict, as in the paper).

    Parameters
    ----------
    path : FIMO tsv (MEME >= 5 format; ``#`` comment lines are skipped). Fi-NeMo tables written by
        ``motif_hit_calling`` use the same columns with p-value ``NA``.
    pvalue_threshold : keep hits with p-value strictly below this. ``None`` keeps every row,
        including NA p-values (required for Fi-NeMo tables).
    sequence_name_parser : maps the ``sequence_name`` column to a DataFrame with at least a ``gene``
        column (plus optional columns such as ``element_class``). Default: sequence_name is the gene.
    use_motif_alt_id : TF = ``motif_alt_id`` for motif ids without ``_`` that have a non-empty alt id
        (JASPAR ``MA0139.1`` / ``CTCF`` -> ``CTCF``). HOCOMOCO ids contain ``_``, so they are unaffected.
        Pass False for Fi-NeMo tables, whose ``motif_alt_id`` is the raw TF-MoDISco pattern id.
    collapse_motif_ids : True (HOCOMOCO): TF = text before the first ``_`` (:func:`collapse_motif_to_tf`).
        False: TF = the motif id as is -- MotifCompendium clusters (``KLF-SP_0``, ``KLF-SP_1``) and
        Fi-NeMo tables are tested per cluster, never pooled by the text before ``_``.

    Returns
    -------
    DataFrame with columns motif_id, tf, pvalue, gene (+ parser columns).
    """
    chunks = []
    wanted = {"motif_id", "sequence_name", "p-value"} | ({"motif_alt_id"} if use_motif_alt_id else set())
    reader = pd.read_csv(
        path, sep="\t", comment="#", usecols=lambda column: column in wanted,
        dtype={"motif_id": str, "motif_alt_id": str, "sequence_name": str}, float_precision="round_trip",
        chunksize=chunksize,
    )
    for chunk in reader:
        chunks.append(chunk if pvalue_threshold is None else chunk[chunk["p-value"] < pvalue_threshold])
    hits = pd.concat(chunks, ignore_index=True).rename(columns={"p-value": "pvalue"})
    if sequence_name_parser is None:
        parsed = pd.DataFrame({"gene": hits["sequence_name"].to_numpy()})
    else:
        parsed = sequence_name_parser(hits["sequence_name"]).reset_index(drop=True)
    hits = pd.concat([hits.drop(columns="sequence_name"), parsed], axis=1)
    tf = collapse_motif_to_tf(hits["motif_id"]) if collapse_motif_ids else hits["motif_id"].astype(str)
    if "motif_alt_id" in hits.columns and collapse_motif_ids:
        alt_id = hits["motif_alt_id"].fillna("").astype(str).str.strip().reset_index(drop=True)
        use_alt = (~hits["motif_id"].astype(str).str.contains("_", regex=False).reset_index(drop=True)
                   & (alt_id != ""))
        tf = tf.where(~use_alt, alt_id)
    hits = hits.drop(columns=[column for column in ["motif_alt_id"] if column in hits.columns])
    hits.insert(1, "tf", tf.to_numpy())
    return hits


def count_hits_per_gene_tf(
    hits: pd.DataFrame,
    genes: Optional[Iterable[str]] = None,
    excluded_element_classes: Iterable[str] = (),
) -> pd.DataFrame:
    """Gene x TF hit-count matrix (rows: genes with >= 1 hit; columns: TFs with >= 1 hit).

    Every hit row counts once, so enhancer hits are summed over all elements linked to a gene.

    Parameters
    ----------
    hits : output of :func:`read_fimo_hits` (needs columns gene, tf; element_class if filtering).
    genes : restrict to these genes (e.g. expressed / cNMF genes) before counting.
    excluded_element_classes : drop hits in elements of these classes (paper: ``promoter`` for ABC).
    """
    keep = np.ones(len(hits), dtype=bool)
    excluded_element_classes = list(excluded_element_classes)
    if excluded_element_classes:
        keep &= ~hits["element_class"].isin(excluded_element_classes).to_numpy()
    if genes is not None:
        keep &= hits["gene"].isin(set(genes)).to_numpy()
    kept = hits.loc[keep, ["gene", "tf"]]
    counts = kept.groupby(["gene", "tf"], sort=True).size().unstack("tf", fill_value=0)
    counts.columns.name = None
    counts.index.name = "gene"
    return counts.sort_index(axis=1)


# ---------------------------------------------------------------------------
# Program genes
# ---------------------------------------------------------------------------

def select_top_program_genes(gene_spectra_score: pd.DataFrame, n_top: int = 300) -> pd.DataFrame:
    """Top ``n_top`` genes per program by score (descending; ties keep input gene order; NaN last).

    Parameters
    ----------
    gene_spectra_score : programs x genes (cNMF ``gene_spectra_score`` orientation).

    Returns
    -------
    Long DataFrame with columns program, gene, rank (1-based).
    """
    rows = []
    for program, scores in gene_spectra_score.iterrows():
        values = scores.to_numpy(dtype=float)
        # stable descending sort with NaN last == dplyr::arrange(desc(x))
        order = np.argsort(np.where(np.isnan(values), np.inf, -values), kind="stable")[:n_top]
        rows.append(pd.DataFrame({
            "program": program, "gene": scores.index[order], "rank": np.arange(1, len(order) + 1),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def adjust_pvalues_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN ignored (n = non-NaN count), as R ``p.adjust``."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    finite = ~np.isnan(pvalues)
    p = pvalues[finite]
    n = p.size
    if n == 0:
        return adjusted
    order = np.argsort(p)[::-1]
    scaled = p[order] * n / np.arange(n, 0, -1)
    adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(scaled))
    result = np.empty(n)
    result[order] = adjusted_sorted
    adjusted[finite] = result
    return adjusted


def welch_ttest_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """Vectorized two-sided Welch t-test (R ``t.test(a, b)``) from group means, variances, sizes.

    Returns (t, df, pvalue). Where R's ``t.test`` would stop with "data are essentially constant"
    (stderr < 10 * eps * max(|mean_a|, |mean_b|)) or a group has < 2 values, returns NaN.
    """
    mean_a, var_a, n_a, mean_b, var_b, n_b = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (mean_a, var_a, n_a, mean_b, var_b, n_b)))
    with np.errstate(divide="ignore", invalid="ignore"):
        se2_a = var_a / n_a
        se2_b = var_b / n_b
        stderr = np.sqrt(se2_a + se2_b)
        df = (se2_a + se2_b) ** 2 / (se2_a ** 2 / (n_a - 1) + se2_b ** 2 / (n_b - 1))
        t = (mean_a - mean_b) / stderr
        pvalue = 2.0 * stats.t.sf(np.abs(t), df)
    constant = stderr < 10 * np.finfo(float).eps * np.maximum(np.abs(mean_a), np.abs(mean_b))
    invalid = constant | (n_a < 2) | (n_b < 2)
    for x in (t, df, pvalue):
        x[invalid] = np.nan
    return t, df, pvalue


def test_motif_enrichment_ttest(
    counts: pd.DataFrame,
    program_genes: pd.DataFrame,
    element_type: str,
) -> pd.DataFrame:
    """Welch t-test of per-gene TF hit counts, program genes vs background (t-test method of Schnitzler et al. 2024).

    Parameters
    ----------
    counts : gene x TF counts from :func:`count_hits_per_gene_tf`; its rows are the universe
        (genes with >= 1 hit, restricted to expressed genes).
    program_genes : long DataFrame with columns program, gene (e.g. :func:`select_top_program_genes`).
        Program genes absent from ``counts`` (0 hits) are dropped.
    element_type : label for the ``element_type`` column (e.g. ``promoter`` / ``enhancer``).

    Returns
    -------
    Long DataFrame (``RESULT_COLUMNS``), one row per program x TF, BH FDR across all rows.
    """
    universe = counts.index
    count_matrix = counts.to_numpy(dtype=float)                     # genes x TFs
    programs = pd.unique(program_genes["program"])
    membership = np.zeros((len(universe), len(programs)), dtype=float)  # genes x programs
    gene_position = pd.Series(np.arange(len(universe)), index=universe)
    for j, program in enumerate(programs):
        genes = program_genes.loc[program_genes["program"] == program, "gene"]
        positions = gene_position.reindex(pd.unique(genes)).dropna().astype(int).to_numpy()
        membership[positions, j] = 1.0

    n_total = float(len(universe))
    total_sum = count_matrix.sum(axis=0)                             # TFs
    total_sumsq = (count_matrix ** 2).sum(axis=0)
    n_program = membership.sum(axis=0)[:, None]                      # programs x 1
    program_sum = membership.T @ count_matrix                        # programs x TFs
    program_sumsq = membership.T @ count_matrix ** 2
    n_background = n_total - n_program
    background_sum = total_sum - program_sum
    background_sumsq = total_sumsq - program_sumsq

    def mean_and_var(total, total_sq, n):
        # (n * sum(x^2) - sum(x)^2) is exact for integer counts, so var is correctly rounded
        with np.errstate(divide="ignore", invalid="ignore"):
            return total / n, (n * total_sq - total ** 2) / (n * (n - 1))

    mean_program, var_program = mean_and_var(program_sum, program_sumsq, n_program)
    mean_background, var_background = mean_and_var(background_sum, background_sumsq, n_background)
    _, _, pvalue = welch_ttest_from_moments(
        mean_program, var_program, n_program, mean_background, var_background, n_background)
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = mean_program / mean_background

    n_tfs = count_matrix.shape[1]
    result = pd.DataFrame({
        "program": np.repeat(programs, n_tfs),
        "element_type": element_type,
        "tf": np.tile(counts.columns.to_numpy(), len(programs)),
        "pvalue": pvalue.ravel(),
        "fdr": adjust_pvalues_bh(pvalue.ravel()),
        "enrichment": enrichment.ravel(),
        "n_program_genes_tested": np.repeat(n_program[:, 0], n_tfs).astype(int),
        "n_background_genes": np.repeat(n_background[:, 0], n_tfs).astype(int),
        "mean_count_program": mean_program.ravel(),
        "mean_count_background": mean_background.ravel(),
    })
    return result[RESULT_COLUMNS]


def pearson_pvalues(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p-value of a correlation coefficient with ``n`` observations (t with n-2 df, as scipy)."""
    r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / ((1.0 - r) * (1.0 + r)))
    return 2.0 * stats.t.sf(np.abs(t), n - 2)


def test_motif_enrichment_correlation(
    counts: pd.DataFrame,
    gene_spectra_score: pd.DataFrame,
    element_type: str,
    correlation: str = "pearson",
    universe_genes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Correlation of per-gene TF hit counts with each program's loading vector (loading-correlation variant).

    Parameters
    ----------
    counts : gene x TF counts from :func:`count_hits_per_gene_tf`.
    gene_spectra_score : programs x genes loading matrix (cNMF ``gene_spectra_score``).
    element_type : label for the ``element_type`` column.
    correlation : ``pearson`` or ``spearman`` (Pearson on average ranks).
    universe_genes : genes to correlate over. Default: every gene of ``gene_spectra_score`` (genes
        with no hit count 0, as in ``enrichment_motif.py``). Genes missing from the loadings are dropped.

    Returns
    -------
    Long DataFrame (``RESULT_COLUMNS``), one row per program x TF; ``enrichment`` = correlation
    coefficient, ``n_program_genes_tested`` = genes correlated over, ``n_background_genes`` = 0, mean
    count columns = mean count over those genes (the same value for every program).
    """
    if correlation not in ("pearson", "spearman"):
        raise ValueError(f"correlation must be pearson or spearman, got {correlation!r}")
    genes = pd.Index(gene_spectra_score.columns if universe_genes is None else list(universe_genes))
    genes = genes[genes.isin(gene_spectra_score.columns)].unique()
    count_matrix = counts.reindex(index=genes, fill_value=0).to_numpy(dtype=float)       # genes x TFs
    loading_matrix = gene_spectra_score.loc[:, genes].to_numpy(dtype=float).T             # genes x programs
    if correlation == "spearman":
        count_matrix = stats.rankdata(count_matrix, axis=0)
        loading_matrix = stats.rankdata(loading_matrix, axis=0)
    n = len(genes)

    def standardize(matrix):
        centered = matrix - matrix.mean(axis=0)
        norm = np.sqrt((centered ** 2).sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            return centered / np.where(norm > 0, norm, np.nan)

    r = standardize(loading_matrix).T @ standardize(count_matrix)                         # programs x TFs
    pvalue = pearson_pvalues(r, n)
    programs = gene_spectra_score.index.to_numpy()
    n_tfs = count_matrix.shape[1]
    mean_count = counts.reindex(index=genes, fill_value=0).to_numpy(dtype=float).mean(axis=0)
    result = pd.DataFrame({
        "program": np.repeat(programs, n_tfs),
        "element_type": element_type,
        "tf": np.tile(counts.columns.to_numpy(), len(programs)),
        "pvalue": pvalue.ravel(),
        "fdr": adjust_pvalues_bh(pvalue.ravel()),
        "enrichment": r.ravel(),
        "n_program_genes_tested": n,
        "n_background_genes": 0,
        "mean_count_program": np.tile(mean_count, len(programs)),
        "mean_count_background": np.tile(mean_count, len(programs)),
    })
    return result[RESULT_COLUMNS]


def flag_significant(results: pd.DataFrame, fdr_threshold: float = 0.05, method: str = "ttest") -> pd.DataFrame:
    """Add boolean ``significant`` = fdr < threshold and enrichment > :data:`SIGNIFICANCE_MIN_ENRICHMENT`
    of ``method`` (t-test: mean ratio > 1, the paper definition; correlation: coefficient > 0)."""
    if method not in SIGNIFICANCE_MIN_ENRICHMENT:
        raise ValueError(f"method must be one of {ENRICHMENT_METHODS}, got {method!r}")
    results = results.copy()
    results["significant"] = ((results["fdr"] < fdr_threshold)
                              & (results["enrichment"] > SIGNIFICANCE_MIN_ENRICHMENT[method]))
    return results
