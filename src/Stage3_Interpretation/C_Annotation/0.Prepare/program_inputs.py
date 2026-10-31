"""Loaders of the Stage 1 / Stage 2 outputs the annotation starts from: the cNMF MuData (spectra, usage, condition
labels), the GO enrichment table and the per-condition perturbation association tables, plus the per-program gene
uniqueness (TF-IDF) and usage-per-condition statistics. Used by prepare_programs.py and build_confounder_screens.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


# loaders
def _check_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def load_gene_spectra(prog_adata, loadings_key="loadings", gene_names_key="var_names"):
    """programs x genes spectra from varm[loadings_key], gene names from uns[gene_names_key]."""
    if loadings_key not in prog_adata.varm:
        raise KeyError(
            f"varm['{loadings_key}'] not found (have {list(prog_adata.varm)})"
        )
    if gene_names_key not in prog_adata.uns:
        raise KeyError(
            f"uns['{gene_names_key}'] not found (have {list(prog_adata.uns)})"
        )
    loadings = np.asarray(prog_adata.varm[loadings_key])
    genes = np.asarray(prog_adata.uns[gene_names_key]).astype(str)
    if loadings.shape[1] != len(genes):
        raise ValueError(
            f"varm['{loadings_key}'] has {loadings.shape[1]} genes but "
            f"uns['{gene_names_key}'] has {len(genes)} names"
        )
    return pd.DataFrame(loadings, index=prog_adata.var_names.astype(int), columns=genes)


def load_cell_usage(prog_adata):
    """cells x programs usage matrix."""
    X = prog_adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return pd.DataFrame(
        X, index=prog_adata.obs_names, columns=prog_adata.var_names.astype(int)
    )


def load_condition(prog_adata, categorical_key="sample"):
    """Cell condition labels and their sorted unique levels (cells with a missing label are
    kept as NaN, so groupby leaves them out of the per-condition stats)."""
    if categorical_key not in prog_adata.obs:
        raise KeyError(
            f"obs['{categorical_key}'] not found (have {list(prog_adata.obs.columns)})"
        )
    raw = prog_adata.obs[categorical_key]
    n_missing = int(raw.isna().sum())
    if n_missing:
        print(
            f"  [warn] {n_missing:,} cells have no obs['{categorical_key}'] label; left out of condition stats"
        )
    labels = (
        raw.astype(object).where(raw.notna()).map(lambda x: x if pd.isna(x) else str(x))
    )
    return labels, sorted(str(c) for c in labels.dropna().unique())


def load_GO(path):
    go_df = pd.read_csv(_check_file(path), sep="\t")
    go_df["program_name"] = go_df["program_name"].astype(int)
    return go_df


def load_perturbation(path):
    assoc_df = pd.read_csv(_check_file(path), sep="\t")
    assoc_df["program_name"] = assoc_df["program_name"].astype(int)
    return assoc_df


def perturbation_paths(path_base, conditions):
    """One '<path_base>_<COND>.txt' file per mdata condition.

    Raises if any condition's file is missing, or if the number of files matching
    '<path_base>_*.txt' differs from the number of conditions.
    """
    base = Path(path_base)
    paths = {cond: base.parent / f"{base.name}_{cond}.txt" for cond in conditions}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No perturbation file for {len(missing)} of {len(conditions)} "
            f"conditions: {missing}"
        )
    found = sorted(base.parent.glob(f"{base.name}_*.txt"))
    if len(found) != len(conditions):
        extra = sorted(set(map(str, found)) - set(map(str, paths.values())))
        raise ValueError(
            f"{len(found)} perturbation files match '{base.name}_*.txt' but mdata has "
            f"{len(conditions)} conditions {conditions}; unmatched files: {extra}"
        )
    return paths


def compute_uniqueness(spectra, membership_top=300):
    """TF-IDF uniqueness, same formula as gpi.enrichment.add_global_uniqueness_scores:

        Score x log((n_programs + 1) / (n_programs_containing_gene + 1))

    A gene "belongs" to a program when it is in that program's top ``membership_top``
    genes (on the full matrix every gene is in every program and the IDF would be 0).
    Computed over ALL programs so a program subset scores identically to a full run.
    Returns a programs x genes frame, NaN outside each program's membership set.
    """
    ranks = spectra.rank(axis=1, ascending=False, method="first")
    member = ranks <= membership_top
    n_prog = spectra.shape[0]
    n_containing = member.sum(axis=0).astype(float)
    idf = np.log((n_prog + 1.0) / (n_containing + 1.0))
    return (spectra * idf).where(member)


def compute_program_condition_stats(usage, cond_labels):
    """Per program, per condition: % of cells with usage > 0 and mean usage.

    Returns {pid: {"top_condition": cond with highest mean usage,
                   "per_condition": {cond: {"pct_cells_expressed", "mean_usage"}}}}.
    """
    grouped = usage.groupby(cond_labels.values)
    mean = grouped.mean()  # compuate program mean usage in condition
    pct = (usage > 0).groupby(
        cond_labels.values
    ).mean() * 100  # compuate program used in % cell in condition
    stats = {}
    for pid in usage.columns:
        stats[pid] = {
            "top_condition": str(mean[pid].idxmax()),
            "per_condition": {
                str(cond): {
                    "pct_cells_expressed": round(float(pct.at[cond, pid]), 1),
                    "mean_usage": float(f"{mean.at[cond, pid]:.3g}"),
                }
                for cond in mean.index
            },
        }
    return stats
