"""Write one self-contained HTML viewer of every labeled program (no external files or libraries).

Reads '<info_dir>/P<k>.json' (AGeneTic bundle) and '<label_dir>/P<k>.json' (program_label.py) and writes one
file: a top bar (search, previous / next, light / dark theme), a rail of programs grouped by the
pre-annotation family label, and one program at a time. The content is the same as program_site.py's pages
(it shares their selection helpers); the layout follows the single-file annotation viewer.

Per program, in page order: the name and one-sentence function; program activity per condition
(program_specificity); top genes (outlined = driving gene) and top unique (distinctive) genes; driving genes
when 2.Evidence_curation/query_driver_gene.py has run for the current label (verdict, why, supporting
papers, other literature searched); significant regulators (an
inline SVG volcano per condition and a regulator x condition log2FC table, with --perturbation_path_base;
then the top regulators given to the annotator); regulator-gene interactions (an interactive Cytoscape.js graph,
inlined, of the top regulators x top and unique genes; edges = every OmniPath interaction found between them by
search_OmniPath.py, plus paper-qa literature relations, read from the bundle's gene_interactions block or, when
it has none, from --interaction_dir); TF motif enrichment (with --motif_enrichment_path: per
element type and motif source, the top significant motif families with logos and candidate TFs, selected
by motif_selection.py); confounder rule-outs, layered interpretation and modules when
program_interpretation.py has run for the current label; top GO terms; the pre-annotation screen when
0.Prepare/build_confounder_screens.py has run. A regulator with negative log2FC on knockdown is an
activator (knocking it down lowers the program); positive is a repressor.
Programs without a label file are skipped. Without --perturbation_path_base / --motif_enrichment_path only
the standard library is needed (pandas otherwise).
"""
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from html_report import (CYTOSCAPE_PATH, _js_safe, gene_card, gene_entries, gene_name, literature_edge,  # noqa: E402
                         load_bundles, omnipath_edge, select_regulators)
from program_site import (CELL_TYPE_TEXT, PUBMED, driver_block, driver_records, family_name,  # noqa: E402
                          gene_paper_meta, load_labels, other_paper_records, paper_records)

TEMPLATE_PATH = HERE / "program_viewer_template.html"
NO_SCREEN, NO_FAMILY = "Not screened", "No family label"


def paper(title, text, meta, same=False):
    """One paper as data: the link rule of program_site.paper_li (PubMed, else DOI, else URL)."""
    pmid = meta.get("pmid")
    url = f"{PUBMED}{pmid}/" if pmid else f"https://doi.org/{meta['doi']}" if meta.get("doi") else meta.get("url")
    return {"title": title, "text": text or "", "pmid": str(pmid or ""), "year": str(meta.get("year") or ""),
            "url": url or "", "same": bool(same)}


def other_papers(gene, shown, paper_meta, bundle):
    gf, lit = other_paper_records(gene, shown, paper_meta, bundle)
    return {"gene_function": [paper(t, text, m) for t, text, m in gf],
            "literature": [paper(t, text, m, same) for t, text, m, same in lit]}


def driver_data(dg, paper_meta, bundle, min_score=5):
    """Driving genes (the explained genes, with supporting and other papers) and how many genes were searched."""
    genes, others = driver_records(dg)

    def why(a):
        j = a.get("judge") or {}
        return j.get("why") or j.get("reason") or a.get("short_excerpt") or ""

    driving = []
    for g in genes:
        a = dg["genes"][g]
        records, shown = paper_records(a, paper_meta.get(g, {}), min_score)
        driving.append({"gene": g, "verdict": a.get("verdict") or "", "why": why(a),
                        "cell_type": CELL_TYPE_TEXT.get(a.get("cell_type_evidence"), ""),
                        "supporting": [paper(t, f, m) for t, f, m in records],
                        "other": other_papers(g, shown, paper_meta.get(g, {}), bundle)})
    return {"driving": driving, "n_searched": len(genes) + len(others),
            "judge_model": (dg.get("params") or {}).get("judge_model") or ""}


def screen_data(bundle):
    """Family label and chromosome enrichment from build_confounder_screens.py (numbers as screen_section)."""
    cs = bundle.get("confounder_screens")
    if not cs:
        return None
    lab, sc, prm = cs["label"], cs["screens"], cs.get("params", {})
    families = []
    for d in lab.get("family_details", []):
        fam = d["family"]
        hit = sc["symbol_families"].get(fam) or sc["marker_sets"].get(fam) or {}
        families.append({"name": family_name(fam), "primary": fam == lab.get("primary_family"),
                         "n_overlap": str(hit.get("n_overlap")), "expected": str(hit.get("expected")),
                         "p_value": str(hit.get("p_value")), "genes": hit.get("genes", [])})
    pos = sc.get("positional", {})
    chrom = None
    if pos.get("assessable"):
        chrom = {k: str(pos.get(k)) for k in ("top_chromosome", "genes_on_top_chromosome", "n_genes_located",
                                              "expected_on_top_chromosome", "fold_enrichment", "binomial_p")}
        chrom["top_chromosome_genes"] = pos.get("top_chromosome_genes", [])
        chrom["p"] = float(pos["binomial_p"])
        chrom["flag"] = bool(lab.get("positional_flag"))
    return {"n_top": sc.get("n_top_genes_screened"), "families": families, "chrom": chrom,
            "min_genes": prm.get("min_genes", 3), "max_p": prm.get("max_p", 1e-3),
            "all_genes": sc.get("top_genes") or []}


def rail_group(bundle):
    cs = bundle.get("confounder_screens")
    if not cs:
        return NO_SCREEN
    fam = cs["label"].get("primary_family")
    return family_name(fam) if fam and fam != "none" else NO_FAMILY


def load_associations(base, conditions, args):
    """{condition: table} of the perturbation association files that exist."""
    import pandas as pd
    out = {}
    for cond in conditions:
        path = f"{base}_{cond}.txt"
        if not Path(path).exists():
            print(f"  [warn] no association file {path}; condition {cond} skipped")
            continue
        df = pd.read_csv(path, sep="\t")
        df[args.program_col] = df[args.program_col].astype(str)
        df[args.target_col] = df[args.target_col].astype(str)
        out[cond] = df
    return out


def regulator_plots(pid, tables, names, index, args):
    """Volcano per condition, compact (regulator index, log2FC x100, -log10 adj p x100, significant), with
    the significance rule of html_Program_QC_plots._build_volcano (adj p <= p_value and log2FC past the up /
    down thresholds); and the regulators with adj p < p_value in any condition (as _build_heatmap), log2FC
    and adj p in every condition, ordered by number of significant conditions then best adj p."""
    volcano, grid = [], {}
    for ci, (cond, df) in enumerate(tables.items()):
        sub = df[df[args.program_col] == str(pid)]
        v = {"condition": cond, "g": [], "x": [], "y": [], "s": []}
        for gene, fc, q in zip(sub[args.target_col], sub[args.log2fc_key], sub["adj_pval"]):
            if fc is None or q is None or math.isnan(fc) or math.isnan(q):
                continue
            if gene not in index:
                index[gene] = len(names)
                names.append(gene)
            sig = q <= args.p_value and (fc >= args.up_thred_log or fc <= args.down_thred_log)
            v["g"].append(index[gene])
            v["x"].append(round(fc * 100))
            v["y"].append(round(-math.log10(max(q, 1e-300)) * 100))
            v["s"].append(int(sig))
            grid.setdefault(gene, [None] * len(tables))[ci] = [round(float(fc), 4), bool(q < args.p_value), float(q)]
        volcano.append(v)
    rows = [{"gene": g, "cells": cells} for g, cells in grid.items() if any(c and c[1] for c in cells)]
    rows.sort(key=lambda r: (-sum(bool(c and c[1]) for c in r["cells"]), min(c[2] for c in r["cells"] if c)))
    return volcano, rows


def attach_interactions(bundles, interaction_dir):
    """Copy gene_interactions from '<interaction_dir>/P<k>.json' into the bundles that have none (the search_OmniPath.py
    output folder can differ from --info_dir). Returns the programs still without one."""
    missing = []
    for pid, bundle in bundles.items():
        if bundle.get("gene_interactions"):
            continue
        path = Path(interaction_dir) / f"P{pid}.json"
        gi = json.loads(path.read_text()).get("gene_interactions") if path.exists() else None
        if gi:
            bundle["gene_interactions"] = gi
        else:
            missing.append(f"P{pid}")
    return missing


def interaction_network(bundle, regs, genes, uniq):
    """Nodes = the shown top regulators, top genes and unique genes; edges = every found OmniPath pair and paper-qa
    relation between a regulator and one of these genes (html_report's edge records). None without gene_interactions."""
    gi = bundle.get("gene_interactions") or {}
    omnipath, literature = gi.get("OmniPath"), gi.get("Literature")
    if not omnipath and not literature:
        return None
    reg_names = [r["gene"] for r in regs]
    kind = {g: "gene" for g in genes}
    for g in uniq:
        kind.setdefault(g, "distinctive")
    nodes = [{"id": f"n_{r['gene']}", "label": r["gene"], "kind": "regulator", "log2fc": r["log2fc"],
              "adj_p": "" if r.get("adj_pval") is None else f"{r['adj_pval']:.2g}", "condition": r["condition"],
              "in_program": r["gene"] in kind} for r in regs]
    for g, k in kind.items():
        if g not in reg_names:   # a regulator that is also a shown gene is drawn once, as a regulator
            nodes.append({"id": f"n_{g}", "label": g, "kind": k, "rank": (genes if k == "gene" else uniq).index(g) + 1})

    included = [p for p in ((omnipath or {}).get("pairs") or {}).values()
                if p["regulator"] in reg_names and p["gene"] in kind]
    edges = [omnipath_edge(p) for p in included if p["found"]]
    edges += [literature_edge(p) for p in ((literature or {}).get("pairs") or {}).values()
              if p.get("source") == "paper-qa" and p.get("category")
              and p["regulator"] in reg_names and p["gene"] in kind]
    return {"nodes": nodes, "edges": edges, "has_omnipath": omnipath is not None,
            "has_literature": literature is not None,
            "n_tested": len(included), "n_found": sum(p["found"] for p in included)}


def program_motifs(pid, motif_tables):
    """motif_selection.select_program_motifs for this program; None without motif tables."""
    if not motif_tables:
        return None
    from motif_selection import select_program_motifs
    return select_program_motifs(pid, *motif_tables)


def load_interpretation(interpretation_dir, pid, label):
    """program_interpretation.py output when it was made for this label, else None."""
    path = Path(interpretation_dir) / f"P{pid}.json"
    if not path.exists():
        return None
    it = json.loads(path.read_text())
    if it.get("label") != label["program_name"]:
        print(f"  [warn] P{pid}: interpretation was made for \"{it.get('label')}\", not the current label; not shown")
        return None
    return {k: it[k] for k in ("confounders", "interpretation", "modules", "coherence", "model")}


def program_data(pid, bundle, label, args, tables, names, index, motif_tables=None):
    cards = {g: gene_card(e) for g, e in gene_entries(bundle).items()}
    dg = driver_block(bundle, label)
    genes = [gene_name(e) for e in bundle.get("program_genes", [])][:args.top_gene]
    uniq = [gene_name(e) for e in bundle.get("distinctive_genes", [])]
    regs = select_regulators(bundle, args.top_regulator)
    screen = screen_data(bundle)
    if dg and any((g.get("judge") or {}).get("error") for g in dg.get("genes", {}).values() if g.get("judge")):
        print(f"  [warn] P{pid}: driving-gene judge verdicts still pending; section not shown")
        dg = None
    drivers = driver_data(dg, gene_paper_meta(bundle), bundle) if dg else None
    shown_genes = set(genes) | set(uniq) | {r["gene"] for r in regs}
    if drivers:
        shown_genes |= {d["gene"] for d in drivers["driving"]}
    if screen:
        shown_genes |= {g for f in screen["families"] for g in f["genes"]} | set(screen["all_genes"])
        shown_genes |= set((screen["chrom"] or {}).get("top_chromosome_genes") or [])
    interpretation = load_interpretation(args.interpretation_dir, pid, label)
    if interpretation:
        shown_genes |= {g for s in interpretation["interpretation"].values() if s
                        for g in s["support_genes"] + s["support_regulators"]}
        shown_genes |= {g for m in interpretation["modules"] for g in m["genes"]}
    volcano, grid = regulator_plots(pid, tables, names, index, args) if tables else ([], [])
    cs = bundle.get("confounder_screens") or {}
    return {
        "id": pid, "label": label["program_name"], "sentence": label["function_sentence"],
        "group": rail_group(bundle), "tag": "positional" if (cs.get("label") or {}).get("positional_flag") else "",
        "top_genes": genes, "distinctive": uniq,
        "genes": {g: {"name": cards[g].get("name") or "", "entrez": str(cards[g].get("entrez") or ""),
                      "symbol": cards[g].get("symbol") or ""} for g in shown_genes if g in cards},
        "regulators": [{**{k: r[k] for k in ("gene", "log2fc", "condition")},
                        "adj_p": "" if r.get("adj_pval") is None else f"{r['adj_pval']:.2g}"} for r in regs],
        "go": [re.sub(r"\s*\(GO:\d+\)", "", t) for t in bundle.get("GO", [])],
        "screen": screen, "drivers": drivers,
        "volcano": volcano, "grid": grid, "n_sig_by_condition": [sum(v["s"]) for v in volcano],
        "network": interaction_network(bundle, regs, genes, uniq),
        "activity": (bundle.get("program_specificity") or {}).get("per_condition") or {},
        "motifs": program_motifs(pid, motif_tables),
        "interpretation": interpretation,
        "label_model": label.get("model", ""),
        "source": f"{Path(args.info_dir).name}/P{pid}.json",
    }


def build_parser():
    p = argparse.ArgumentParser(description="Write one self-contained HTML viewer of every labeled program.")

    # IO
    p.add_argument("--info_dir", required=True, help="Folder of P<k>.json bundles (Literature_info_extended_PerturbNMF_Info recommended).")
    p.add_argument("--label_dir", default=None, help="program_label.py output. Default: <info_dir>/../Program_label.")
    p.add_argument("--interpretation_dir", default=None, help="program_interpretation.py output (confounder rule-outs, layered interpretation, modules); shown when present. Default: <info_dir>/../Program_interpretation.")
    p.add_argument("--interaction_dir", default=None, help="search_OmniPath.py output folder, read for the programs whose --info_dir bundle has no gene_interactions block; adds the regulator-gene interaction graph. Default: <info_dir>/../Gene_info_extended_PerturbNMF_Info.")
    p.add_argument("--out", default=None, help="Output HTML. Default: <info_dir>/../Report/program_viewer.html.")

    # context info
    p.add_argument("--dataset_name", required=True, help="Dataset label shown in the top bar.")
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids, space separated (e.g. 1 2 3). Default: every P<k>.json in --info_dir.")

    # content
    p.add_argument("--top_gene", type=int, default=15, help="Top-loading program genes shown.")
    p.add_argument("--top_regulator", type=int, default=6, help="Regulators shown (best adj_pval over conditions).")
    p.add_argument("--membership_top", type=int, default=None, help="Top-loaded genes per program used for GO enrichment and unique genes (0.Prepare membership_top); shown in the card captions.")

    # regulator-program association plots (optional)
    p.add_argument("--perturbation_path_base", default=None, help="Perturbation association results without the _<condition>.txt suffix (e.g. Evaluation/<K>_<thr>/<K>_perturbation_association_results); adds the volcano + log2FC table. Default: no plots.")
    p.add_argument("--conditions", nargs="+", default=None, help="Conditions, space separated (files <base>_<condition>.txt). Default: the bundles' conditions.")
    p.add_argument("--target_col", default="target_name", help="Perturbed-gene column of the association files.")
    p.add_argument("--program_col", default="program_name", help="Program column of the association files (values match the P<k> number).")
    p.add_argument("--log2fc_key", default="log2FC", help="log2FC column of the association files (approx_log2FC for CRT results).")
    p.add_argument("--p_value", type=float, default=0.05, help="adj p cutoff for the volcano colors and the log2FC table regulators.")
    p.add_argument("--down_thred_log", type=float, default=-0.05, help="Volcano log2FC threshold for down.")
    p.add_argument("--up_thred_log", type=float, default=0.05, help="Volcano log2FC threshold for up.")

    # TF motif enrichment (optional; Stage 2 run_motif_enrichment.py outputs)
    p.add_argument("--motif_enrichment_path", default=None, help="Stage 2 motif enrichment table (e.g. Evaluation/<K>_<thr>/<K>_motif_enrichment.txt); adds the TF motif enrichment card. The run's <K>_motif_enrichment_config.yml next to it gives the test (t-test / correlation, top genes). Default: no motif card.")
    p.add_argument("--candidate_tfs_path", default=None, help="Stage 2 candidate TF table (<K>_candidate_tfs.txt); adds the candidate TFs under each motif family.")
    p.add_argument("--motif_logos_path", default=None, help="Stage 2 motif logos (<K>_motif_logos.json); adds a sequence logo per motif.")
    p.add_argument("--motif_source_label", default="", help="Free-text provenance of the motif results, shown under the motif card (e.g. motif database, promoter window, enhancer links).")
    return p


def main():
    args = build_parser().parse_args()
    bundles = load_bundles(args.info_dir, args.programs)
    label_dir = Path(args.label_dir or Path(args.info_dir).resolve().parent / "Program_label")
    args.interpretation_dir = args.interpretation_dir or Path(args.info_dir).resolve().parent / "Program_interpretation"
    labels = load_labels(label_dir, bundles)
    skipped = [f"P{p}" for p in bundles if p not in labels]
    if skipped:
        print(f"  [warn] no label in {label_dir} for {skipped} (run program_label.py); skipped")
    if not labels:
        raise SystemExit("No program has a label; nothing to write.")

    interaction_dir = Path(args.interaction_dir or Path(args.info_dir).resolve().parent / "Gene_info_extended_PerturbNMF_Info")
    no_net = attach_interactions({pid: bundles[pid] for pid in labels}, interaction_dir)
    if no_net:
        print(f"  [warn] no gene_interactions for {no_net} in --info_dir or {interaction_dir} "
              f"(run search_OmniPath.py); no interaction graph for them")

    first = bundles[next(iter(labels))]
    conditions = args.conditions or first.get("conditions") or []
    tables = load_associations(args.perturbation_path_base, conditions, args) if args.perturbation_path_base else {}
    motif_tables = None
    if args.motif_enrichment_path:   # pandas only when motif results are shown
        from motif_selection import read_motif_logos, read_motif_tables, read_motif_test, shown_logos
        motif_tables = read_motif_tables(args.motif_enrichment_path, args.candidate_tfs_path)
    names, index, programs = [], {}, {}
    for pid in labels:
        programs[pid] = program_data(pid, bundles[pid], labels[pid], args, tables, names, index, motif_tables)
        net = programs[pid]["network"]
        print(f"  P{pid}: {labels[pid]['program_name']}"
              + (f" ({len(net['edges'])} regulator-gene interactions)" if net else ""))

    counts = {}
    for p in programs.values():
        counts[p["group"]] = counts.get(p["group"], 0) + 1
    last = {NO_FAMILY: 1, NO_SCREEN: 2}
    groups = sorted(counts, key=lambda g: (last.get(g, 0), -counts[g], g))
    meta = {
        "title": args.dataset_name,
        "conditions": list(tables) or conditions, "groups": groups, "regulator_names": names,
        "p_value": args.p_value, "up": args.up_thred_log, "down": args.down_thred_log,
        "membership_top": args.membership_top, "built": datetime.now().strftime("%Y-%m-%d"),
        "organism": first.get("organism") or "human",
        "motif_test": read_motif_test(args.motif_enrichment_path) if motif_tables else None,
        "motif_logos": (shown_logos(programs.values(), read_motif_logos(args.motif_logos_path))
                        if motif_tables and args.motif_logos_path else {}),
        "motif_source": args.motif_source_label,
    }
    out = Path(args.out or Path(args.info_dir).resolve().parent / "Report" / "program_viewer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    cyto = ""
    if any(p["network"] for p in programs.values()):   # inline Cytoscape.js only when a graph is drawn
        lib = CYTOSCAPE_PATH.read_text()
        if "</script" in lib.lower():
            raise ValueError(f"{CYTOSCAPE_PATH} contains '</script'; cannot inline it")
        cyto = f"<script>{lib}</script>"
    # small placeholders first, then the data and library blobs (never re-scanned)
    html = (TEMPLATE_PATH.read_text()
            .replace("__PAGE_TITLE__", args.dataset_name.replace("&", "&amp;").replace("<", "&lt;"))
            .replace("__META_JSON__", _js_safe(json.dumps(meta, separators=(",", ":"))))
            .replace("__PROGRAMS_JSON__", _js_safe(json.dumps(programs, separators=(",", ":")))))
    html = html.replace("__CYTOSCAPE_TAG__", cyto, 1)
    out.write_text(html)
    print(f"[done] {len(programs)} program(s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
