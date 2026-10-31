"""
Helpers for pulling matrices out of AnnData / MuData, shaping CRT inputs, and
formatting CRT results.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_from_adata(adata: Any, key: str) -> Any:
    """
    Try common AnnData containers, then fall back to item access.

    Args:
        adata: AnnData object to search.
        key: "X" for adata.X, otherwise a name to look up in obsm, layers,
            obsp, uns, then obs.

    Returns:
        The stored value (array, sparse matrix, DataFrame column, or uns entry).

    Raises:
        KeyError: If `key` is not found in any container.
    """
    if key == "X" and hasattr(adata, "X"):
        return adata.X
    if hasattr(adata, "obsm") and key in adata.obsm:
        return adata.obsm[key]
    if hasattr(adata, "layers") and key in adata.layers:
        return adata.layers[key]
    if hasattr(adata, "obsp") and key in adata.obsp:
        return adata.obsp[key]
    if hasattr(adata, "uns") and key in adata.uns:
        return adata.uns[key]
    if hasattr(adata, "obs") and key in adata.obs:
        return adata.obs[key]
    try:
        return adata[key]
    except Exception as exc:  # pylint: disable=broad-except
        raise KeyError(f"Could not find `{key}` in adata.* containers.") from exc


def get_from_mudata(mdata: Any, key: str, mod: Optional[str] = None) -> Any:
    """
    Look up `key` in a MuData object.

    - If `mod` is given, search only that modality (via `get_from_adata`).
    - `key` may be written as "<mod>:<key>" to target a modality.
    - Otherwise search MuData-level containers first, then each modality in
      order, returning the first hit.

    Args:
        mdata: MuData object to search.
        key: Name to look up, or "<mod>:<key>" to target one modality.
        mod: Optional modality name (e.g. "rna"); restricts the search to
            `mdata.mod[mod]`.

    Returns:
        The stored value from the first container that holds `key`.

    Raises:
        KeyError: If `mod` is not a modality, or `key` is not found anywhere.
    """
    if mod is not None:
        if mod not in mdata.mod:
            raise KeyError(f"Modality `{mod}` not in mdata.mod ({list(mdata.mod)}).")
        return get_from_adata(mdata.mod[mod], key)

    # MuData-level containers (obs columns here are already "<mod>:<col>")
    for container in ("obsm", "obsp", "uns", "obs"):
        if hasattr(mdata, container) and key in getattr(mdata, container):
            return getattr(mdata, container)[key]

    if ":" in key:
        # prefix: modality name; subkey: key inside that modality
        prefix, subkey = key.split(":", 1)
        if prefix in mdata.mod:
            return get_from_adata(mdata.mod[prefix], subkey)

    # m: one modality's AnnData
    for m in mdata.mod.values():
        try:
            return get_from_adata(m, key)
        except KeyError:
            continue

    raise KeyError(
        f"Could not find `{key}` in mdata.* containers or any modality ({list(mdata.mod)})."
    )


def to_csc_matrix(Guide_assignment: Any) -> Tuple[sp.csc_matrix, Optional[List[str]]]:
    """
    Cast guide assignment to CSC (efficient column slicing per guide).

    Args:
        Guide_assignment: cells x guides matrix (DataFrame, sparse, or array-like).

    Returns:
        Guide_assignment as csc_matrix, and guide names if the input was a
        DataFrame (else None).
    """
    if isinstance(Guide_assignment, pd.DataFrame):
        guide_names = list(Guide_assignment.columns)
        return sp.csc_matrix(Guide_assignment.to_numpy()), guide_names
    if sp.issparse(Guide_assignment):
        G = Guide_assignment.tocsc(copy=True)
        G.eliminate_zeros()
        return G, None
    return sp.csc_matrix(np.asarray(Guide_assignment)), None


def load_modality(
    h5mu_path: str, usage_key: str, covariate_cols: Sequence[str] = ()
) -> Any:
    """
    Read one modality of an h5mu and drop cells with a missing value in any
    covariate column.

    Args:
        h5mu_path: MuData file.
        usage_key: modality to read (e.g. "cNMF").
        covariate_cols: obs columns that must be non-missing.

    Returns:
        MuData with only that modality.
    """
    import mudata as md

    adata = md.read_h5ad(h5mu_path, usage_key)
    keep = np.ones(adata.n_obs, dtype=bool)
    for col in covariate_cols:
        if col not in adata.obs:
            raise KeyError(f"Covariate `{col}` not in obs of modality `{usage_key}`.")
        keep &= adata.obs[col].notna().to_numpy()
    if not keep.all():
        print(
            f"cells: {adata.n_obs:,} -> {int(keep.sum()):,} (dropped {int((~keep).sum()):,} with missing covariates)"
        )
        adata = adata[keep].copy()
    return md.MuData({usage_key: adata})


def split_by_condition(
    mdata: Any,
    usage_key: str,
    categorical_key: str,
    conditions: Optional[Sequence[str]] = None,
):
    """
    Split a MuData into one MuData per condition, to run the CRT separately in each.

    Args:
        mdata: MuData (e.g. from load_modality).
        usage_key: modality whose obs holds categorical_key.
        categorical_key: obs column with each cell's condition (e.g. sample, timepoint).
        conditions: conditions to keep, in this order; None uses every non-missing
            level, in order of first appearance.

    Yields:
        (condition, MuData with only that condition's cells and only usage_key).
    """
    import mudata as md

    adata = mdata.mod[usage_key]
    if categorical_key not in adata.obs:
        raise KeyError(f"`{categorical_key}` not in obs of modality `{usage_key}`.")
    labels = adata.obs[categorical_key]
    if conditions is None:
        conditions = list(pd.unique(labels.dropna()))
    for condition in conditions:
        keep = (labels == condition).to_numpy()
        if not keep.any():
            raise ValueError(f"No cells with {categorical_key} == {condition!r}.")
        yield condition, md.MuData({usage_key: adata[keep].copy()})


def adjust_pvalues(pvals: np.ndarray, fdr_method: str = "BH") -> np.ndarray:
    """
    FDR-adjusted p-values over all tests: "BH" (Benjamini-Hochberg) or "StoreyQ"
    (Storey q-value, multipy.fdr.qvalue, as in PerturbNMF's CRT.py).
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    if fdr_method == "BH":
        from statsmodels.stats.multitest import multipletests

        return multipletests(pvals, method="fdr_bh")[1]
    if fdr_method == "StoreyQ":
        import builtins

        if not hasattr(builtins, "xrange"):  # multipy is written for Python 2
            builtins.xrange = range
        from multipy.fdr import qvalue

        return qvalue(pvals, threshold=0.05, verbose=False)[1]
    raise ValueError(f'fdr_method must be "BH" or "StoreyQ"; got {fdr_method!r}.')


def log2fc_from_beta(
    beta: np.ndarray, control_mean: np.ndarray, normalization: str, n_programs: int
) -> np.ndarray:
    """
    Approximate log2 fold change of program usage from the CRT effect size beta.

    clr: beta is a shift in CLR coordinate k; the CLR mean absorbs 1/K of it, so
        ln-FC = beta * K/(K-1) and log2FC = [K/(K-1)] * beta / ln(2).
    row_normalize: beta is the covariate-adjusted difference in usage share, so
        log2FC = log2((control_mean + beta) / control_mean), shares floored at 1e-8.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if normalization == "clr":
        return beta * (n_programs / (n_programs - 1)) / np.log(2)
    tiny = 1e-8
    control_mean = np.asarray(control_mean, dtype=np.float64)
    return np.log2(
        np.maximum(control_mean + beta, tiny) / np.maximum(control_mean, tiny)
    )


# Results table columns (PerturbNMF format, read by Stage 3)
RESULTS_COLUMNS = [
    "target_name",
    "program_name",
    "log2FC",
    "p-value",
    "adj_pval",
    "pval_empirical",
    "adj_pval_empirical",
    "pval_skewnorm",
    "beta",
    "n_treated",
    "n_control",
    "control_mean",
    "n_resampled",
    "propensity_n_iter",
    "propensity_sum_ratio",
]


def format_results_table(
    obj, fdr_method: str = "BH", include_guides: bool = False
) -> pd.DataFrame:
    """
    Long table: one row per tested gene x program, in the PerturbNMF format read by Stage 3.

    Columns:
        target_name: tested gene (fake genes: "null_<i>").
        program_name: program number, 1-based ("1".."K").
        log2FC: approximate log2 fold change of program usage (log2fc_from_beta).
        p-value: final CRT p (pval_skewnorm where the skew-normal was fit, else pval_empirical).
        adj_pval: p-value adjusted across every gene x program test in the table
            (fdr_method "BH" or "StoreyQ").
        pval_empirical: resampling p, (1 + #extreme) / (1 + rounds).
        adj_pval_empirical: pval_empirical adjusted the same way.
        pval_skewnorm: skew-normal p; NaN where not fit.
        beta: CRT effect size (usage share or CLR scale, per obj.normalization).
        n_treated, n_control: cells in the gene's regression.
        control_mean: mean normalized usage in the control cells.
        n_resampled: resampling rounds used.
        propensity_n_iter: lbfgs iterations of the gene's propensity fit (moi="high"); equal to
            max_iter means the fit hit its cap; 0 = intercept-only (no fit); NaN = no propensity model.
        propensity_sum_ratio: sum(p) / n_treated of that fit; 1 at a converged (unpenalized-intercept) fit.
        guides (include_guides=True, for the non-targeting fake genes): the gene's
            guide names, ";"-joined.
    """
    rows = []
    for gene, r in obj.gene_results.items():
        nan = np.full(r.betas.size, np.nan)
        pval_empirical = (
            r.pvalue_empirical if r.pvalue_empirical is not None else r.pvalue
        )
        pval_skewnorm = r.pvalue_skewnorm if r.pvalue_skewnorm is not None else nan
        control_means = r.control_means if r.control_means is not None else nan
        for k, program in enumerate(obj.selected_program_names):
            rows.append(
                (
                    gene,
                    program,
                    r.betas[k],
                    r.pvalue[k],
                    pval_empirical[k],
                    pval_skewnorm[k],
                    r.n_treated,
                    r.n_control,
                    control_means[k],
                    r.n_resampled,
                    r.propensity_n_iter,
                    r.propensity_sum_ratio,
                )
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "target_name",
            "program_name",
            "beta",
            "p-value",
            "pval_empirical",
            "pval_skewnorm",
            "n_treated",
            "n_control",
            "control_mean",
            "n_resampled",
            "propensity_n_iter",
            "propensity_sum_ratio",
        ],
    )
    table["log2FC"] = log2fc_from_beta(
        table["beta"],
        table["control_mean"],
        obj.normalization,
        len(obj.selected_program_names),
    )
    table["adj_pval"] = adjust_pvalues(table["p-value"].to_numpy(), fdr_method)
    table["adj_pval_empirical"] = adjust_pvalues(
        table["pval_empirical"].to_numpy(), fdr_method
    )
    table = table[RESULTS_COLUMNS].copy()
    if include_guides:
        table["guides"] = table["target_name"].map(_guide_string(obj))
    return table


def _guide_string(obj) -> dict:
    """gene -> ";"-joined guide names, for every tested gene."""
    return {
        g: ";".join(obj.guide_names[j] for j in obj.guide_targets[g])
        for g in obj.selected_gene_names
    }


def load_results_table(obj, path: str):
    """
    Fill obj.gene_results from a table written by format_results_table (tab-separated
    .txt / .tsv, or .csv), instead of rerunning conditional_randomized_test
    (--skip_existing in PerturbNMF's CRT.py).

    The table must cover exactly obj.selected_gene_names x obj.selected_program_names;
    if it has a "guides" column (fake genes), each gene's guides must match
    obj.guide_targets, so a cached null built with other settings is rejected.
    Cached results have no treated_obs_idx (stored as an empty array) and no
    skew-normal fit parameters.

    Args:
        obj: CRT_obj with the same genes / programs as the cached run.
        path: table from format_results_table.

    Returns:
        The same CRT_obj with gene_results filled.
    """
    from .dataclass import CRT_gene_result

    sep = "," if path.endswith(".csv") else "\t"
    table = pd.read_csv(path, sep=sep, dtype={"program_name": str})
    genes, programs = list(obj.selected_gene_names), list(obj.selected_program_names)
    if (
        set(table["target_name"]) != set(genes)
        or set(table["program_name"]) != set(programs)
        or len(table) != len(genes) * len(programs)
    ):
        raise ValueError(
            f"{path} does not match this run: {table['target_name'].nunique()} genes x "
            f"{table['program_name'].nunique()} programs ({len(table)} rows) vs {len(genes)} genes x "
            f"{len(programs)} programs expected."
        )
    if "guides" in table:
        cached = (
            table.drop_duplicates("target_name")
            .set_index("target_name")["guides"]
            .to_dict()
        )
        expected = _guide_string(obj)
        mismatched = [g for g in genes if cached[g] != expected[g]]
        if mismatched:
            raise ValueError(
                f"{path}: guides differ for {len(mismatched)} genes (e.g. {mismatched[0]}); "
                "the cached null was built with other settings."
            )

    table = table.set_index(["target_name", "program_name"])
    obj.gene_results = {}
    for gene in genes:
        rows = table.loc[gene].loc[programs]
        pvalue_skewnorm = rows["pval_skewnorm"].to_numpy(dtype=np.float64)
        obj.gene_results[gene] = CRT_gene_result(
            gene=gene,
            n_treated=int(rows["n_treated"].iloc[0]),
            treated_obs_idx=np.empty(0, dtype=np.int32),
            pvalue=rows["p-value"].to_numpy(dtype=np.float64),
            betas=rows["beta"].to_numpy(dtype=np.float64),
            pvalue_skewnorm=pvalue_skewnorm
            if obj.pvalue_method == "skew_normal"
            else None,
            pvalue_empirical=rows["pval_empirical"].to_numpy(dtype=np.float64),
            control_means=rows["control_mean"].to_numpy(dtype=np.float64),
            n_resampled=int(rows["n_resampled"].iloc[0]),
            n_control=int(rows["n_control"].iloc[0]),
            # tables written before these columns existed: NaN
            propensity_n_iter=float(rows["propensity_n_iter"].iloc[0])
            if "propensity_n_iter" in rows
            else np.nan,
            propensity_sum_ratio=(
                float(rows["propensity_sum_ratio"].iloc[0])
                if "propensity_sum_ratio" in rows
                else np.nan
            ),
        )
    return obj
