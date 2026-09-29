import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import mudata as mu
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
        raise KeyError(f"varm['{loadings_key}'] not found (have {list(prog_adata.varm)})")
    if gene_names_key not in prog_adata.uns:
        raise KeyError(f"uns['{gene_names_key}'] not found (have {list(prog_adata.uns)})")
    loadings = np.asarray(prog_adata.varm[loadings_key])
    genes = np.asarray(prog_adata.uns[gene_names_key]).astype(str)
    if loadings.shape[1] != len(genes):
        raise ValueError(f"varm['{loadings_key}'] has {loadings.shape[1]} genes but "
                         f"uns['{gene_names_key}'] has {len(genes)} names")
    return pd.DataFrame(loadings, index=prog_adata.var_names.astype(int), columns=genes)


def load_cell_usage(prog_adata):
    """cells x programs usage matrix."""
    X = prog_adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return pd.DataFrame(X, index=prog_adata.obs_names, columns=prog_adata.var_names.astype(int))


def load_condition(prog_adata, categorical_key="sample"):
    """Cell condition labels and their sorted unique levels (cells with a missing label are
    kept as NaN, so groupby leaves them out of the per-condition stats)."""
    if categorical_key not in prog_adata.obs:
        raise KeyError(f"obs['{categorical_key}'] not found (have {list(prog_adata.obs.columns)})")
    raw = prog_adata.obs[categorical_key]
    n_missing = int(raw.isna().sum())
    if n_missing:
        print(f"  [warn] {n_missing:,} cells have no obs['{categorical_key}'] label; left out of condition stats")
    labels = raw.astype(object).where(raw.notna()).map(lambda x: x if pd.isna(x) else str(x))
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
        raise FileNotFoundError(f"No perturbation file for {len(missing)} of {len(conditions)} "
                                f"conditions: {missing}")
    found = sorted(base.parent.glob(f"{base.name}_*.txt"))
    if len(found) != len(conditions):
        extra = sorted(set(map(str, found)) - set(map(str, paths.values())))
        raise ValueError(f"{len(found)} perturbation files match '{base.name}_*.txt' but mdata has "
                         f"{len(conditions)} conditions {conditions}; unmatched files: {extra}")
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
    mean = grouped.mean()                  # compuate program mean usage in condition
    pct = (usage > 0).groupby(cond_labels.values).mean() * 100  # compuate program used in % cell in condition
    stats = {}
    for pid in usage.columns:
        stats[pid] = {
            "top_condition": str(mean[pid].idxmax()),
            "per_condition": {
                str(cond): {"pct_cells_expressed": round(float(pct.at[cond, pid]), 1),
                            "mean_usage": float(f"{mean.at[cond, pid]:.3g}")}
                for cond in mean.index
            },
        }
    return stats


# selectors
def load_top_gene(spectra, pid, n=15):
    """Top-N genes by loading score (stable sort, ties keep column order)."""
    row = spectra.loc[pid].sort_values(ascending=False, kind="mergesort")
    return row.index[:n].tolist()


def load_top_unique_gene(uniqueness, pid, m=8, exclude=()):
    """Top-M genes by uniqueness, excluding the top-loading genes (disjoint sets)."""
    row = uniqueness.loc[pid].dropna().sort_values(ascending=False, kind="mergesort")
    exclude = set(exclude)
    return [g for g in row.index if g not in exclude][:m]


def load_top_GO(go_df, pid, n=10, fdr=0.05):
    """Top-N significant GO terms (by adjusted p, then Combined Score), with any
    parenthesized text such as the GO ID removed."""
    sub = go_df[(go_df["program_name"] == pid) & (go_df["Adjusted P-value"] < fdr)]
    sub = sub.sort_values(["Adjusted P-value", "Combined Score"],
                          ascending=[True, False], kind="mergesort")
    return [re.sub(r"\s*\([^)]*\)", "", t).strip() for t in sub["Term"].head(n)]


def load_regulator(assoc_df, pid, n=6, fdr=0.05, lfc_col="log2FC"):
    """Top-N significant regulators (adj_pval < fdr), ranked by adj_pval then |lfc_col|."""
    if lfc_col not in assoc_df.columns:
        raise KeyError(f"log2FC column '{lfc_col}' not found (have {list(assoc_df.columns)})")
    sub = assoc_df[(assoc_df["program_name"] == pid) & (assoc_df["adj_pval"] < fdr)].copy()
    sub["_abs_lfc"] = sub[lfc_col].abs()
    sub = sub.sort_values(["adj_pval", "_abs_lfc"], ascending=[True, False], kind="mergesort")
    return [
        {"gene": str(r["target_name"]),
         "log2fc": round(float(r[lfc_col]), 2),
         "adj_pval": float(f"{r['adj_pval']:.3g}")}
        for _, r in sub.head(n).iterrows()
    ]


def load_regulator_gene_pairs(program_genes, distinctive_genes, regulators):
    """Every regulator x (program genes, then distinctive genes): the pair list investigated
    downstream (OmniPath marks validated pairs, the literature step follows up on the rest).

    Regulators are merged across conditions in first-seen order, keeping per-condition stats;
    a regulator is never paired with itself. Keys are 'GENE-REGULATOR', as in OmniPath's pairs.
    """
    reg_stats = {}
    for cond, regs in regulators.items():
        for r in regs:
            reg_stats.setdefault(r["gene"], {})[cond] = {"log2fc": r["log2fc"], "adj_pval": r["adj_pval"]}
    genes = [(g, "program_gene") for g in program_genes] + [(g, "distinctive_gene") for g in distinctive_genes]
    pairs = {}
    for reg, stats in reg_stats.items():
        for gene, category in genes:
            if gene != reg:
                pairs[f"{gene}-{reg}"] = {"gene": gene, "regulator": reg, "gene_category": category,
                                          "regulator_stats": stats}
    return {"n_pairs": len(pairs), "n_regulators": len(reg_stats), "n_genes": len(genes), "pairs": pairs}


# research brief
def build_research_brief(label, organism, cell_type, conditions, has_regulators,
                         top_condition=None):
    """Short instruction referencing the bundle's fields (adapted from GPI bundle.py)."""
    subject = cell_type or organism or "cell"
    role = f"{cell_type} biologist" if cell_type else "cell biologist"
    reg_clause = (
        " and the genes in `perturbation_regulators` (research these the same way as the "
        "program genes)"
        if has_regulators else ""
    )
    lines = [
        f"# Program {label} — {organism} {subject} gene program",
        "",
        f"You are a {role}. Determine the shared biological function of this program's genes.",
        "",
        f"Research the genes in `program_genes` and `distinctive_genes`{reg_clause}, guided by "
        "the enriched GO terms listed in `GO`. Land on 1-3 coherent functional themes, each "
        "supported by several genes and specific retrieved papers.",
    ]
    if has_regulators:
        lines += [
            "",
            "`regulator_gene` lists every regulator x program/distinctive gene pair to investigate: "
            "check which links are validated, and what connects the rest.",
        ]
    if conditions:
        lines += [
            "",
            f"Experimental context: {', '.join(conditions)}. Cite the condition link when the "
            "literature supports it; do not force one.",
        ]
    if top_condition:
        lines += [
            "",
            f"`program_specificity` gives, per condition, the % of cells using this program "
            f"and its mean usage; usage is highest in {top_condition}.",
        ]
    lines += [
        "",
        "Cite ONLY PMIDs/DOIs your tools return — never fabricate an identifier, title, or "
        "quotation. Do not assign the final program label.",
    ]
    return "\n".join(lines)


# bundle
def build_bundle(pid, spectra, uniqueness, go_df, assoc_by_cond, cond_stats, args):
    label = f"P{pid}"
    program_genes = load_top_gene(spectra, pid, args.top_gene)
    distinctive_genes = load_top_unique_gene(uniqueness, pid, args.top_unique_gene,
                                             exclude=program_genes)
    go_terms = load_top_GO(go_df, pid, args.top_GO, args.fdr)
    if not go_terms:
        print(f"  [warn] {label}: no GO term with adj p < {args.fdr}")

    regulators = {}
    for cond, assoc_df in assoc_by_cond.items():
        regs = load_regulator(assoc_df, pid, args.top_regulator, args.fdr, args.log2fc_key)
        if regs:
            regulators[cond] = regs

    conditions = list(assoc_by_cond)
    bundle = {
        "program_id": label,
        "organism": args.organism,
        "cell_type": args.cell_type,
        "conditions": conditions,
        "GO": go_terms,
        "program_genes": program_genes,
        "distinctive_genes": distinctive_genes,
        "program_specificity": cond_stats[pid],
    }
    if regulators:
        bundle["perturbation_regulators"] = regulators
        bundle["regulator_gene"] = load_regulator_gene_pairs(program_genes, distinctive_genes, regulators)
    bundle["research_brief"] = build_research_brief(
        label, args.organism, args.cell_type, conditions, bool(regulators),
        top_condition=cond_stats[pid]["top_condition"])
    return bundle


def build_parser():
    p = argparse.ArgumentParser(description="Extract annotation bundles from PerturbNMF outputs.")

    # IO
    p.add_argument("--mdata_path", required=True, help="Path to the cNMF MuData (.h5mu) file.")
    p.add_argument("--GO_path", required=True, help="GO_term_enrichment.txt.")
    p.add_argument("--perturbation_path_base", required=True, help="Path base of the perturbation association files; '<base>_<COND>.txt' is read for each level of --categorical_key.")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    # keys
    p.add_argument("--data_key", default="rna", help="Key of the gene expression modality in MuData (recorded in meta only).")
    p.add_argument("--prog_key", default="cNMF", help="Key of the cNMF program modality in MuData.")
    p.add_argument("--loadings_key", default="loadings", help="varm key of the program x gene spectra in the program modality.")
    p.add_argument("--gene_name_key", default="var_names", help="uns key of the gene names matching the loadings columns.")
    p.add_argument("--categorical_key", default="sample", help="obs key of the condition labels.")
    p.add_argument("--log2fc_key", default="log2FC", help="log2FC column in the perturbation files (e.g. 'approx_log2FC' for CRT results).")

    # context info
    p.add_argument("--programs", type=int, nargs="+", help="Program ids, space separated (e.g. 1 2 3). Default: all.")
    p.add_argument("--cell_type", default="", help="Cell type, e.g. 'teloHAEC aortic endothelial cell'.")
    p.add_argument("--organism", default="human")

    # select top items of the program
    p.add_argument("--top_gene", type=int, default=15, help="Top-N genes by loading score.")
    p.add_argument("--top_unique_gene", type=int, default=8, help="Top-M genes by uniqueness, excluding the top-loading N.")
    p.add_argument("--membership_top", type=int, default=300, help="Top genes per program counted as members for uniqueness IDF.")

    p.add_argument("--top_GO", type=int, default=10, help="Top GO terms per program.")
    p.add_argument("--top_regulator", type=int, default=6, help="Top significant regulators per program per condition.")

    # significance cutoff shared by every adjusted p-value filter (GO terms, regulators)
    p.add_argument("--fdr", type=float, default=0.05, help="Adjusted p-value cutoff for GO terms and regulators.")
    return p


def main():
    args = build_parser().parse_args()

    # read only the program modality (obs carries the condition labels)
    prog_adata = mu.read_h5ad(str(_check_file(args.mdata_path)), mod=args.prog_key)

    cond_labels, conditions = load_condition(prog_adata, args.categorical_key)
    print(f"[load] {len(conditions)} conditions in obs['{args.categorical_key}']: {conditions}")

    spectra = load_gene_spectra(prog_adata, args.loadings_key, args.gene_name_key)
    print(f"[load] spectra: {spectra.shape[0]} programs x {spectra.shape[1]} genes")

    usage = load_cell_usage(prog_adata)
    print(f"[load] usage: {usage.shape[0]:,} cells x {usage.shape[1]} programs")

    go_df = load_GO(args.GO_path)

    # one perturbation file per mdata condition
    perturb_paths = perturbation_paths(args.perturbation_path_base, conditions)
    assoc_by_cond = {}
    for cond, path in perturb_paths.items():
        assoc_by_cond[cond] = load_perturbation(path)
        print(f"[load] perturbation '{cond}': {len(assoc_by_cond[cond]):,} rows")

    # unique genes and per-condition program usage
    uniqueness = compute_uniqueness(spectra, args.membership_top)
    cond_stats = compute_program_condition_stats(usage, cond_labels)

    # programs to compile
    available = sorted(spectra.index)
    programs = args.programs or available
    missing = sorted(set(programs) - set(available))
    if missing:
        raise ValueError(f"Programs not in spectra (available {available[0]}..{available[-1]}): {missing}")
    programs = list(dict.fromkeys(programs))  # dedupe, keep order

    # make output folder
    out_dir = Path(args.out_dir)
    bundle_dir = out_dir / "PerturbNMF_Info"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # one JSON per program
    labels, n_pairs = [], {}
    for pid in programs:
        bundle = build_bundle(pid, spectra, uniqueness, go_df, assoc_by_cond, cond_stats, args)
        (bundle_dir / f"{bundle['program_id']}.json").write_text(json.dumps(bundle, indent=2))
        regs = bundle.get("perturbation_regulators", {})
        reg_summary = ", ".join(f"{c}={len(regs.get(c, []))}" for c in conditions)
        n_pairs[bundle["program_id"]] = bundle.get("regulator_gene", {}).get("n_pairs", 0)
        print(f"  {bundle['program_id']}: genes={len(bundle['program_genes'])} "
              f"distinctive={len(bundle['distinctive_genes'])} GO={len(bundle['GO'])} "
              f"regulators[{reg_summary}] pairs={n_pairs[bundle['program_id']]} "
              f"top={bundle['program_specificity']['top_condition']}")
        labels.append(bundle["program_id"])

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "mdata_path": str(args.mdata_path),
            "GO_path": str(args.GO_path),
            "perturbation_path_base": str(args.perturbation_path_base),
            "perturbation_path": {c: str(p) for c, p in perturb_paths.items()},
        },
        "keys": {k: getattr(args, k) for k in (
            "data_key", "prog_key", "loadings_key", "gene_name_key", "categorical_key", "log2fc_key")},
        "params": {k: getattr(args, k) for k in (
            "top_gene", "top_unique_gene", "membership_top", "top_GO",
            "top_regulator", "fdr")},
        "organism": args.organism,
        "cell_type": args.cell_type,
        "conditions": conditions,
        "programs": labels,
        "n_regulator_gene_pairs": n_pairs,
        "n_regulator_gene_pairs_total": sum(n_pairs.values()),
    }
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(labels)} bundle(s) -> {bundle_dir}; meta -> {bundle_dir / 'meta.json'}")
    print("Extraction of programs finished")
    return 0


if __name__ == '__main__':
    sys.exit(main())
