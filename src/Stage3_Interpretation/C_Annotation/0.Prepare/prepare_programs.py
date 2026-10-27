"""Create one folder per cNMF program with its tables and plan: the starting point of every later step.

    python prepare_programs.py --mdata_path cNMF_60_0_2.h5mu --GO_path 60_GO_term_enrichment.txt \
        --perturbation_path_base 60_perturbation_association_results --run_dir <out_dir>/<run_name>/Annotation

writes <run_dir>/Programs/meta.json and, per program, <run_dir>/Programs/P<k>/:

    program.json        id, organism, cell type, conditions, per-condition usage (program_specificity), counts
    genes_top300.tsv    rank, gene, loading, distinctive_score (TF-IDF over all programs' top --membership_top)
    regulators_sig.tsv  every regulator with adj_p < --fdr in any condition: regulator, condition, log2fc, adj_p, role
    go_sig.tsv          every GO term with adj_p < --fdr: term, go_id, adj_p, combined_score, overlap, genes
    string_sig.tsv      every STRING term with FDR <= --fdr for the top genes (agent/string-skill), and
    string_genes.json   STRING PPI enrichment and per-gene partners (top genes and regulators), and the links among
                        the regulators (regulator_network: type experimental / curated / predicted, and the
                        regulators' PPI enrichment against the same background)
    literature.jsonl    empty; the literature step appends one row per paper
    plan.md             the program's steps (0.Prepare/program_dir.py), with prepare ticked once its checks pass

A program folder that already exists keeps its plan and later outputs; only the prepare outputs are rewritten
(--overwrite) or the program is skipped. STRING needs network access (about 4 calls per program, 1 s apart);
--skip_string writes an empty string_sig.tsv instead.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import mudata as mu

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "agent" / "string-skill" / "scripts")]
from program_inputs import (_check_file, compute_program_condition_stats, compute_uniqueness,  # noqa: E402
                            load_cell_usage, load_condition, load_gene_spectra, load_GO, load_perturbation,
                            perturbation_paths)
from program_dir import mark_done, new_plan, read_plan, sha256, write_plan, write_tsv  # noqa: E402

GO_ID = re.compile(r"\((GO:\d+)\)")


def top_genes(spectra, uniqueness, pid, n):
    row = spectra.loc[pid].sort_values(ascending=False, kind="mergesort").head(n)
    uniq = uniqueness.loc[pid]
    return [{"rank": i, "gene": str(g), "loading": f"{v:.6g}",
             "distinctive_score": "" if uniq.isna()[g] else f"{uniq[g]:.6g}"}
            for i, (g, v) in enumerate(row.items(), 1)]


def significant_regulators(assoc_by_cond, pid, fdr, lfc_col):
    rows = []
    for cond, df in assoc_by_cond.items():
        sub = df[(df["program_name"] == pid) & (df["adj_pval"] < fdr)].copy()
        sub["_abs"] = sub[lfc_col].abs()
        sub = sub.sort_values(["adj_pval", "_abs"], ascending=[True, False], kind="mergesort")
        rows += [{"regulator": str(r["target_name"]), "condition": cond, "log2fc": f"{r[lfc_col]:.4g}",
                  "adj_p": f"{r['adj_pval']:.3g}", "role": "activator" if r[lfc_col] < 0 else "repressor"}
                 for _, r in sub.iterrows()]
    return rows


def significant_go(go_df, pid, fdr):
    sub = go_df[(go_df["program_name"] == pid) & (go_df["Adjusted P-value"] < fdr)]
    sub = sub.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False], kind="mergesort")
    out = []
    for _, r in sub.iterrows():
        m = GO_ID.search(str(r["Term"]))
        out.append({"term": re.sub(r"\s*\([^)]*\)", "", str(r["Term"])).strip(), "go_id": m.group(1) if m else "",
                    "adj_p": f"{r['Adjusted P-value']:.3g}", "combined_score": f"{r['Combined Score']:.4g}",
                    "overlap": f"{int(r['overlap_numerator'])}/{int(r['overlap_denominator'])}",
                    "genes": ",".join(g for g in str(r["Genes"]).split(";") if g)})
    return out


def build_parser():
    p = argparse.ArgumentParser(description="One folder per program: tables, empty literature file and plan.")
    p.add_argument("--mdata_path", required=True, help="cNMF MuData (.h5mu).")
    p.add_argument("--GO_path", required=True, help="GO_term_enrichment.txt of the Stage 2 evaluation.")
    p.add_argument("--perturbation_path_base", required=True, help="Path base of the perturbation association files; '<base>_<COND>.txt' is read for each condition.")
    p.add_argument("--run_dir", required=True, help="Annotation run folder; programs go to <run_dir>/Programs/P<k>/.")
    p.add_argument("--prog_key", default="cNMF", help="Program modality in the MuData.")
    p.add_argument("--loadings_key", default="loadings", help="varm key of the program x gene spectra.")
    p.add_argument("--gene_name_key", default="var_names", help="uns key of the gene names of the spectra.")
    p.add_argument("--categorical_key", default="sample", help="obs key of the condition labels.")
    p.add_argument("--log2fc_key", default="log2FC", help="log2FC column of the perturbation files (e.g. approx_log2FC for CRT).")
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids. Default: all.")
    p.add_argument("--cell_type", default="", help="Cell type, e.g. 'aortic endothelial cell'.")
    p.add_argument("--organism", default="human")
    p.add_argument("--n_genes", type=int, default=300, help="Top genes per program written to genes_top300.tsv and sent to STRING.")
    p.add_argument("--membership_top", type=int, default=300, help="Top genes per program counted as members for the distinctive (TF-IDF) score.")
    p.add_argument("--fdr", type=float, default=0.05, help="Adjusted p-value cutoff for regulators, GO terms and STRING terms.")
    p.add_argument("--skip_string", action="store_true", help="No STRING queries; string_sig.tsv is written empty.")
    p.add_argument("--string_background", default="spectra", choices=["spectra", "genome"], help="STRING enrichment background: every gene in the spectra, or STRING's genome.")
    p.add_argument("--string_required_score", type=int, default=700, help="Min STRING combined score x1000 for partners (700 = high confidence).")
    p.add_argument("--taxid", type=int, default=9606, help="NCBI taxonomy id for STRING (9606 = human).")
    p.add_argument("--overwrite", action="store_true", help="Rewrite the prepare outputs of programs that already have a folder (their plan and later outputs are kept).")
    return p


def main():
    args = build_parser().parse_args()
    prog = mu.read_h5ad(str(_check_file(args.mdata_path)), mod=args.prog_key)
    cond_labels, conditions = load_condition(prog, args.categorical_key)
    spectra = load_gene_spectra(prog, args.loadings_key, args.gene_name_key)
    usage = load_cell_usage(prog)
    go_df = load_GO(args.GO_path)
    perturb_paths = perturbation_paths(args.perturbation_path_base, conditions)
    assoc = {c: load_perturbation(path) for c, path in perturb_paths.items()}
    if args.log2fc_key not in next(iter(assoc.values())).columns:
        sys.exit(f"--log2fc_key {args.log2fc_key!r} is not a column of the perturbation files")
    uniqueness = compute_uniqueness(spectra, args.membership_top)
    cond_stats = compute_program_condition_stats(usage, cond_labels)
    print(f"[load] {spectra.shape[0]} programs x {spectra.shape[1]} genes; conditions {conditions}")

    available = sorted(spectra.index)
    programs = list(dict.fromkeys(args.programs or available))
    missing = sorted(set(programs) - set(available))
    if missing:
        sys.exit(f"programs not in the spectra: {missing}")
    root = Path(args.run_dir) / "Programs"
    root.mkdir(parents=True, exist_ok=True)

    todo = [pid for pid in programs if args.overwrite or not (root / f"P{pid}" / "plan.md").exists()]
    for pid in sorted(set(programs) - set(todo)):
        print(f"  P{pid}: folder exists; skipped (use --overwrite)")

    string_ids, background = None, None
    if todo and not args.skip_string:
        from string_enrichment import map_string_ids
        names = [str(g) for pid in todo for g in spectra.loc[pid].sort_values(ascending=False).index[:args.n_genes]]
        names += [str(r) for df in assoc.values() for r in df["target_name"].dropna().unique()]
        if args.string_background == "spectra":
            names += [str(g) for g in spectra.columns]
        string_ids = map_string_ids(names, args.taxid, root / "STRING_id_map.json")
        if args.string_background == "spectra":
            background = sorted({string_ids[g] for g in map(str, spectra.columns) if g in string_ids})
        print(f"[STRING] {len(string_ids):,} names mapped; background "
              f"{len(background):,} genes" if background else "[STRING] background: genome")

    for pid in todo:
        d = root / f"P{pid}"
        d.mkdir(exist_ok=True)
        genes = top_genes(spectra, uniqueness, pid, args.n_genes)
        regs = significant_regulators(assoc, pid, args.fdr, args.log2fc_key)
        gos = significant_go(go_df, pid, args.fdr)
        write_tsv(d / "genes_top300.tsv", genes, ("rank", "gene", "loading", "distinctive_score"))
        write_tsv(d / "regulators_sig.tsv", regs, ("regulator", "condition", "log2fc", "adj_p", "role"))
        write_tsv(d / "go_sig.tsv", gos, ("term", "go_id", "adj_p", "combined_score", "overlap", "genes"))
        n_string = 0
        if string_ids is not None:
            from string_enrichment import enrich_program, write_tables
            candidates = {g["gene"]: "program gene" for g in genes}
            for r in regs:
                candidates.setdefault(r["regulator"], "regulator")
            result = enrich_program([g["gene"] for g in genes], string_ids, background, candidates,
                                    {"taxid": args.taxid, "max_fdr": args.fdr,
                                     "required_score": args.string_required_score},
                                    regulators=[r["regulator"] for r in regs])
            write_tables(result, d)
            n_string = len(result["terms"])
        else:
            write_tsv(d / "string_sig.tsv", [], ("category", "term", "description", "fdr", "p_value", "n_genes",
                                                 "n_background", "genes"))
        if not (d / "literature.jsonl").exists():
            (d / "literature.jsonl").write_text("")
        (d / "program.json").write_text(json.dumps({
            "program_id": f"P{pid}", "organism": args.organism, "cell_type": args.cell_type, "conditions": conditions,
            "program_specificity": cond_stats[pid], "fdr": args.fdr,
            "counts": {"genes": len(genes), "regulators_sig": len({r['regulator'] for r in regs}),
                       "go_sig": len(gos), "string_sig": n_string}}, indent=2))
        if not (d / "plan.md").exists():
            write_plan(d, new_plan(f"P{pid}"))
        errors = mark_done(d, "prepare", by="prepare_programs.py")
        print(f"  P{pid}: regulators {len({r['regulator'] for r in regs})}, GO {len(gos)}, STRING {n_string}"
              + (f"  [prepare FAILED: {errors}]" if errors else ""))

    (root / "meta.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"mdata_path": str(Path(args.mdata_path).resolve()), "GO_path": str(Path(args.GO_path).resolve()),
                   "perturbation_path": {c: str(Path(p).resolve()) for c, p in perturb_paths.items()},
                   "sha256": {"GO_path": sha256(args.GO_path),
                              **{f"perturbation_{c}": sha256(p) for c, p in perturb_paths.items()}}},
        "keys": {k: getattr(args, k) for k in ("prog_key", "loadings_key", "gene_name_key", "categorical_key",
                                               "log2fc_key")},
        "params": {k: getattr(args, k) for k in ("n_genes", "membership_top", "fdr", "skip_string",
                                                 "string_background", "string_required_score", "taxid")},
        "organism": args.organism, "cell_type": args.cell_type, "conditions": conditions,
        "programs": [f"P{p}" for p in available]}, indent=2))
    print(f"[done] {len(todo)} program folder(s) -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
